"""
generate_demo_dialogue.py — Custom Demo Dialogue Generator (Advisor Demo)
==========================================================================
Builds a controlled-comparison demo: the SAME 3 client utterances (text +
your recorded voice) are shown to two therapist conditions:

  - Baseline    : transcript only (gpt-4o-mini, temp 0.7)
  - Dissonance  : transcript + text/speech VA + vocal descriptors + delta + flag
                  (gpt-4o-mini, temp 0.7)

Then a blind GPT-4o judge (temp 0.0) evaluates both sessions, so the demo
shows real scores for THIS dialogue.

Extraction logic is copied verbatim from run_dissonance_multibackbone.py
(vad-bert normalization, WavLM logit order [arousal, dominance, valence],
librosa descriptor bins, Russell circumplex, per-dimension dissonance rule).

Usage:
  1. Record U1.wav, U2.wav, U3.wav into  demo_custom/raw/
     (script.json with the utterance texts is auto-created on first run)
  2. python generate_demo_dialogue.py --check    <- extraction only, free
  3. python generate_demo_dialogue.py            <- full: + therapists + judge
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

# ------------------------------------------------------------------
# Paths / config
# ------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
CUSTOM_DIR = BASE / "demo_custom"
RAW_DIR = CUSTOM_DIR / "raw"
AUDIO_DIR = CUSTOM_DIR / "audio"
SCRIPT_JSON = RAW_DIR / "script.json"

THERAPIST_MODEL = "gpt-4o-mini"
EVALUATOR_MODEL = "gpt-4o"

DEFAULT_SCRIPT = [
    # U1 - FLAG target: positive words, read them TENSE (strained, controlled).
    "Honestly, work is going really well right now, I feel completely fine about everything.",
    # U2 - aligned: genuinely tired.
    "But when I actually stop to think about it, I feel exhausted all the time.",
    # U3 - payoff: quiet admission.
    "Maybe I have just been pretending to be much stronger than I really am.",
]

# ------------------------------------------------------------------
# Prompts (verbatim canon from the thesis pipeline)
# ------------------------------------------------------------------
BASELINE_THERAPIST_SYSTEM = """
You are "Luna", a warm, empathetic CBT therapist.

Your goals:
- Understand the client's thoughts, emotions, and behaviors.
- Validate their feelings without dismissing or catastrophizing.
- Gently use CBT techniques (identify automatic thoughts, examine evidence,
  explore alternative perspectives, plan small experiments).

Rules:
- Reply in 2–3 sentences.
- Do NOT mention any numeric scores, analytics, or models.
- End most responses with an open question that invites reflection.
"""

BASELINE_USER_TEMPLATE = """
Client just said:
"{client_text}"

Please write your next therapist response to the client.
"""

DISS_THERAPIST_SYSTEM = """
You are "Luna", a CBT therapist with access to both the client's words and
an analysis of their voice, including how their vocal delivery matches or
mismatches those words.

For each client message you receive:
- Text-based emotion: Valence_text, Arousal_text (from -1 to +1)
- Voice-based emotion: Valence_speech, Arousal_speech (from -1 to +1)
- Vocal prosodic descriptors: pitch level, loudness, speaking rate
- Dissonance: delta_valence = speech - text, delta_arousal = speech - text

Interpretation guidelines:
- Large |delta| means the client's tone and words are pulling in different directions.
- Vocal descriptors provide additional context beyond the VA numbers.
- When dissonance is large, gently explore the mismatch.

Your job:
- Respond with empathy, using CBT principles.
- When dissonance is high, reflect what might be "under the surface".
- Ask curious, non-judgmental questions.

Important:
- NEVER mention numbers, "dissonance", or "analysis".
- Speak only in natural language.
- Reply in 2–3 sentences.
- Still follow CBT principles.
"""

DISS_USER_TEMPLATE = """
Client just said:
"{client_text}"

Estimates from analysis:
- Text emotion:
    - Valence_text: {val_t:.2f}
    - Arousal_text: {aro_t:.2f}
- Voice emotion:
    - Valence_speech: {val_s:.2f}
    - Arousal_speech: {aro_s:.2f}
- Vocal cues (prosodic features):
    {vocal_descriptors}
- Dissonance (speech - text):
    - delta_valence: {delta_v:.2f}
    - delta_arousal: {delta_a:.2f}

Overall dissonance flag: {is_dissonant}

Write your next therapist response. If the mismatch is large, gently explore.
Do not mention any numbers.
"""

# Evaluation prompts (VERBATIM from evaluation/evaluate_multibackbone_final.py —
# the exact RQ2 protocol used for the thesis's multi-backbone results, so the
# custom demo is judged under an identical protocol.)
EVAL_SYSTEM_PROMPT = """You are an expert psychological evaluator specializing in Cognitive Behavioral Therapy (CBT) and therapeutic alliance. Your task is to analyze a full counseling session transcript and score the therapist's performance based on the evaluation frameworks used in the MIRROR paper (arXiv:2504.13211v2).

CRITICAL EVALUATION FOCUS:
- You must assess the therapist's ability to perceive "Emotional Subtext"—the underlying feelings that may not be explicitly stated in the client's words but are hinted at through context, tone, or provided emotional metadata.
- A high-performing therapist should move beyond surface-level reflection and identify "Latent Concerns" to facilitate deeper discovery.
You must evaluate the entire session as a whole, not just individual turns.
"""

EVAL_USER_PROMPT = """Please evaluate this therapy session. Pay close attention to how the therapist navigates the gap between what the client SAYS and what the client FEELS

[Session Transcript]
{conversation}
[/Session Transcript]

1. **Therapist Skills (0-6):** Understanding, Interpersonal Effectiveness
2. **Client Alliance (1-5):** Affective Bond
3. **CTRS (0-6 each):** Collaboration, Guided Discovery, Focus, Strategy

Return EXACT JSON:
{{
  "therapist_skills": {{"understanding": 0, "interpersonal_effectiveness": 0}},
  "client_alliance": {{"affective_bond": 0}},
  "ctrs": {{"collaboration": 0, "guided_discovery": 0, "focus": 0, "strategy": 0}},
  "reasoning": "...",
  "comparative_advantage": "..."
}}
"""

# ------------------------------------------------------------------
# Lazy globals (loaded once)
# ------------------------------------------------------------------
device = None
vad_tokenizer = None
vad_model = None
_wavlm = None
_wavlm_sr = None
_wavlm_mean = None
_wavlm_std = None
_client = None


def setup_openai_client():
    global _client
    if _client is not None:
        return _client
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ").strip()
    _client = OpenAI()
    return _client


def chat_once(system_prompt: str, user_prompt: str, model: str,
              temperature: float = 0.7, json_mode: bool = False) -> str:
    client = setup_openai_client()
    kwargs = dict(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=1500 if json_mode else 512,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Phase A: extraction (logic copied from run_dissonance_multibackbone.py)
# ------------------------------------------------------------------
def load_models():
    global device, vad_tokenizer, vad_model, _wavlm, _wavlm_sr, _wavlm_mean, _wavlm_std
    if vad_model is not None:
        return
    import torch
    from transformers import (AutoModelForAudioClassification,
                              AutoModelForSequenceClassification,
                              AutoTokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading vad-bert: RobroKools/vad-bert ...")
    vad_tokenizer = AutoTokenizer.from_pretrained("RobroKools/vad-bert")
    vad_model = AutoModelForSequenceClassification.from_pretrained(
        "RobroKools/vad-bert").to(device)
    vad_model.eval()

    print("Loading WavLM: 3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes ...")
    _wavlm = AutoModelForAudioClassification.from_pretrained(
        "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
        trust_remote_code=True).to(device)
    _wavlm.eval()
    _wavlm_sr = _wavlm.config.sampling_rate
    _wavlm_mean = _wavlm.config.mean
    _wavlm_std = _wavlm.config.std


def _to_minus1_1(x, xmin=1.0, xmax=5.0):
    return float(2 * (x - xmin) / (xmax - xmin) - 1.0)


def get_text_VA(text: str) -> Tuple[float, float]:
    import torch
    enc = vad_tokenizer(text, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = vad_model(**enc)
    v_raw, a_raw, d_raw = out.logits.cpu().numpy()[0].tolist()
    return _to_minus1_1(v_raw), _to_minus1_1(a_raw)


def get_speech_VA(wav_path: Path) -> Tuple[float, float]:
    import librosa
    import soundfile as sf
    import torch
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _wavlm_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_wavlm_sr)
    audio = (audio - _wavlm_mean) / (_wavlm_std + 1e-6)
    wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, wavs.shape[1], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = _wavlm(wavs, mask)
    logits = pred.cpu().numpy()[0].astype(float)
    aro, dom, val = float(logits[0]), float(logits[1]), float(logits[2])
    return (2.0 * val - 1.0), (2.0 * aro - 1.0)


def get_vocal_descriptors(wav_path: Path, client_text: str) -> str:
    import librosa
    y, sr_native = librosa.load(str(wav_path), sr=None)
    duration = len(y) / sr_native
    pitches, _ = librosa.piptrack(y=y, sr=sr_native)
    pitch_vals = pitches[pitches > 0]
    pitch_mean = float(pitch_vals.mean()) if len(pitch_vals) > 0 else 0.0
    rms = librosa.feature.rms(y=y)
    rms_mean = float(rms.mean())
    word_count = len(client_text.split())
    speech_rate = word_count / duration if duration > 0 else 1.0

    if pitch_mean < 100: pitch_desc = "very low pitch"
    elif pitch_mean < 150: pitch_desc = "low pitch"
    elif pitch_mean < 200: pitch_desc = "moderate pitch"
    elif pitch_mean < 250: pitch_desc = "high pitch"
    else: pitch_desc = "very high pitch"

    if rms_mean < 0.02: loud_desc = "very quiet"
    elif rms_mean < 0.05: loud_desc = "soft-spoken"
    elif rms_mean < 0.10: loud_desc = "moderate volume"
    elif rms_mean < 0.15: loud_desc = "loud"
    else: loud_desc = "very loud"

    if speech_rate < 2.0: rate_desc = "slow speech"
    elif speech_rate < 3.0: rate_desc = "moderate-paced speech"
    elif speech_rate < 4.0: rate_desc = "fast speech"
    else: rate_desc = "very rapid speech"

    return f"{pitch_desc}, {loud_desc}, {rate_desc}"


def get_discrete_emotion(val: float, aro: float) -> str:
    distance = math.sqrt(val ** 2 + aro ** 2)
    if distance < 0.15:
        return "Neutral"
    angle = math.degrees(math.atan2(aro, val))
    if angle < 0:
        angle += 360
    centers = [(0, "Happy"), (45, "Excited"), (90, "Tense"), (135, "Angry"),
               (180, "Sad"), (225, "Depressed"), (270, "Calm"), (315, "Content")]
    closest = min(centers, key=lambda c: min(abs(angle - c[0]), 360 - abs(angle - c[0])))
    return closest[1]


# ------------------------------------------------------------------
# Setup helpers
# ------------------------------------------------------------------
def ensure_script_json() -> list:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not SCRIPT_JSON.exists():
        SCRIPT_JSON.write_text(json.dumps(DEFAULT_SCRIPT, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        print(f"Created {SCRIPT_JSON}")
        print(">> Record U1.wav, U2.wav, U3.wav into demo_custom/raw/ reading these lines.")
    return json.loads(SCRIPT_JSON.read_text(encoding="utf-8"))


def find_wavs(n_turns: int) -> list:
    wavs = []
    for i in range(1, n_turns + 1):
        p = RAW_DIR / f"U{i}.wav"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}\nRecord the utterance and save it as U{i}.wav "
                f"(any sample rate; mono or stereo).")
        wavs.append(p)
    return wavs


def extract_all(script: list, wavs: list) -> list:
    load_models()
    meta = []
    for i, (text, wav) in enumerate(zip(script, wavs), start=1):
        print(f"[A] Turn {i}: extracting ...")
        val_t, aro_t = get_text_VA(text)
        val_s, aro_s = get_speech_VA(wav)
        voc = get_vocal_descriptors(wav, text)
        te = get_discrete_emotion(val_t, aro_t)
        se = get_discrete_emotion(val_s, aro_s)
        dv, da = val_s - val_t, aro_s - aro_t
        is_dis = (abs(dv) >= 0.5) or (abs(da) >= 0.5)
        meta.append({
            "turn": i, "client": text,
            "val_t": val_t, "aro_t": aro_t, "val_s": val_s, "aro_s": aro_s,
            "text_emotion": te, "speech_emotion": se,
            "vocal_descriptors": voc,
            "delta_valence": dv, "delta_arousal": da,
            "is_dissonant": is_dis, "emotion_mismatch": te != se,
        })
    return meta


def print_calibration(meta: list):
    print("\n" + "=" * 74)
    print("CALIBRATION TABLE (extraction only — no LLM calls made)")
    print("=" * 74)
    hdr = f"{'T':<3}{'Text VA':<16}{'Voice VA':<16}{'Text emo':<11}{'Voice emo':<12}{'dV':<7}{'dA':<7}{'FLAG'}"
    print(hdr)
    print("-" * 74)
    for m in meta:
        tv = f"({m['val_t']:+.2f},{m['aro_t']:+.2f})"
        sv = f"({m['val_s']:+.2f},{m['aro_s']:+.2f})"
        flag = "🚩 YES" if m["is_dissonant"] else "no"
        print(f"{m['turn']:<3}{tv:<16}{sv:<16}{m['text_emotion']:<11}{m['speech_emotion']:<12}"
              f"{m['delta_valence']:+.2f}  {m['delta_arousal']:+.2f}  {flag}")
    print("-" * 74)
    flagged = any(m["is_dissonant"] for m in meta)
    if flagged:
        print("✅ At least one turn flags — good to proceed with the full run.")
    else:
        print("⚠️  No turn flagged yet. For the demo you WANT U1 to flag.")
        print("   Tips: re-record U1 MORE tense/strained/faster while keeping the")
        print("   positive words, or make the text more positive, e.g.:")
        print('   "I\'m honestly doing really great at work, everything is completely under control."')
    print("=" * 74 + "\n")


# ------------------------------------------------------------------
# Phase B: therapist generation (stateless, matches thesis pipeline)
# ------------------------------------------------------------------
def generate_therapists(meta: list) -> tuple:
    baseline_turns, diss_turns = [], []
    for m in meta:
        t = m["turn"]
        print(f"[B] Turn {t}: baseline therapist ...")
        b_reply = chat_once(
            BASELINE_THERAPIST_SYSTEM,
            BASELINE_USER_TEMPLATE.format(client_text=m["client"]),
            model=THERAPIST_MODEL, temperature=0.7)
        baseline_turns.append({
            "turn": t, "client": m["client"], "therapist": b_reply,
            "condition": "baseline",
            "val_t": m["val_t"], "aro_t": m["aro_t"], "text_emotion": m["text_emotion"],
        })
        print(f"[B] Turn {t}: dissonance therapist ...")
        d_reply = chat_once(
            DISS_THERAPIST_SYSTEM,
            DISS_USER_TEMPLATE.format(
                client_text=m["client"], val_t=m["val_t"], aro_t=m["aro_t"],
                val_s=m["val_s"], aro_s=m["aro_s"],
                vocal_descriptors=m["vocal_descriptors"],
                delta_v=m["delta_valence"], delta_a=m["delta_arousal"],
                is_dissonant=m["is_dissonant"]),
            model=THERAPIST_MODEL, temperature=0.7)
        diss_turns.append({
            "turn": t, "client": m["client"], "therapist": d_reply,
            "condition": "dissonance_aware_therapist",
            **{k: m[k] for k in ("val_t", "aro_t", "val_s", "aro_s", "text_emotion",
                                  "speech_emotion", "vocal_descriptors",
                                  "delta_valence", "delta_arousal",
                                  "is_dissonant", "emotion_mismatch")},
        })
    return baseline_turns, diss_turns


# ------------------------------------------------------------------
# Phase C: blind GPT-4o evaluation
# ------------------------------------------------------------------
def transcript_of(turns: list) -> str:
    lines = []
    for t in turns:
        lines.append(f"Client: {t['client']}")
        lines.append(f"Therapist: {t['therapist']}")
    return "\n".join(lines)


def evaluate_session(turns: list, label: str) -> dict:
    print(f"[C] Evaluating {label} session (blind judge) ...")
    raw = chat_once(EVAL_SYSTEM_PROMPT,
                    EVAL_USER_PROMPT.format(conversation=transcript_of(turns)),
                    model=EVALUATOR_MODEL, temperature=0.0, json_mode=True)
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ------------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------------
def save_outputs(baseline_turns: list, diss_turns: list, scores: dict, wavs: list):
    for i, wav in enumerate(wavs, start=1):
        shutil.copy2(wav, AUDIO_DIR / f"U{i}.wav")
    for t in baseline_turns:
        t["audio_path"] = str(AUDIO_DIR / f"U{t['turn']}.wav")
    for t in diss_turns:
        t["audio_path"] = str(AUDIO_DIR / f"U{t['turn']}.wav")

    b_path = CUSTOM_DIR / "baseline_custom.jsonl"
    d_path = CUSTOM_DIR / "dissonance_custom.jsonl"
    with open(b_path, "w", encoding="utf-8") as f:
        for t in baseline_turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(d_path, "w", encoding="utf-8") as f:
        for t in diss_turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    (CUSTOM_DIR / "scores_custom.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved:\n  {b_path}\n  {d_path}\n  {CUSTOM_DIR / 'scores_custom.json'}")


def print_score_summary(scores: dict):
    def total(s):
        ts, ca, ct = s["therapist_skills"], s["client_alliance"], s["ctrs"]
        return (ts["understanding"] + ts["interpersonal_effectiveness"]
                + ca["affective_bond"] + ct["collaboration"]
                + ct["guided_discovery"] + ct["focus"] + ct["strategy"])
    if "Baseline" in scores and "Dissonance" in scores:
        bt, dt = total(scores["Baseline"]), total(scores["Dissonance"])
        print(f"\nJudge totals — Baseline: {bt}  |  Dissonance: {dt}  |  Δ = {dt - bt:+}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Custom demo dialogue generator")
    ap.add_argument("--check", action="store_true",
                    help="Phase A only (extraction + calibration table, no LLM calls)")
    ap.add_argument("--eval-only", action="store_true",
                    help="Phase C only: re-judge the existing demo_custom/*.jsonl sessions "
                         "(no extraction, no therapist regeneration)")
    args = ap.parse_args()

    if args.eval_only:
        b_path = CUSTOM_DIR / "baseline_custom.jsonl"
        d_path = CUSTOM_DIR / "dissonance_custom.jsonl"
        if not (b_path.exists() and d_path.exists()):
            print("No existing sessions found. Run the full generator first.")
            sys.exit(1)
        def _read(p):
            return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        baseline_turns, diss_turns = _read(b_path), _read(d_path)
        scores = {
            "Baseline": evaluate_session(baseline_turns, "Baseline"),
            "Dissonance": evaluate_session(diss_turns, "Dissonance"),
        }
        (CUSTOM_DIR / "scores_custom.json").write_text(
            json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {CUSTOM_DIR / 'scores_custom.json'}")
        print_score_summary(scores)
        return

    script = ensure_script_json()
    print(f"Script ({len(script)} utterances):")
    for i, s in enumerate(script, start=1):
        print(f"  U{i} ({len(s.split())} words): {s}")

    wavs = find_wavs(len(script))
    meta = extract_all(script, wavs)
    print_calibration(meta)

    if args.check:
        print("--check mode: stopping before LLM calls.")
        return

    baseline_turns, diss_turns = generate_therapists(meta)
    scores = {
        "Baseline": evaluate_session(baseline_turns, "Baseline"),
        "Dissonance": evaluate_session(diss_turns, "Dissonance"),
    }
    save_outputs(baseline_turns, diss_turns, scores, wavs)
    print_score_summary(scores)
    print("\nDone. Launch demo_luna.py and pick '★ Custom Demo (Advisor)'.")


if __name__ == "__main__":
    main()
