#!/usr/bin/env python3
"""
LLM Inference Timing — measures every step EXCEPT Zonos TTS synthesis.
1 backbone, 1 dialogue, per condition. Outputs .md report.

Usage:
    python inference_timing.py --model qwen --condition dissonance --start 1 --end 1
"""
import argparse
import os
import sys
import json
import math
import time
import statistics
from pathlib import Path
from typing import Tuple, Dict, List

import torch
import numpy as np
import soundfile as sf
import librosa
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForAudioClassification

# =====================================================
# CLI
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--condition", type=str, required=True,
                    choices=["baseline", "emotion", "multimodal", "dissonance"])
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--end", type=int, default=1)
args = parser.parse_args()

# =====================================================
# Platform-aware paths
# =====================================================
import platform
if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

INFERENCE_DIR = BASE_DIR / "dissonance" / "inference_time"
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_BASE = BASE_DIR / "dissonance" / "multibackbone"
SHARED_SEEDS_PATH = OUTPUT_BASE / "shared_seeds.json"
if not SHARED_SEEDS_PATH.exists():
    raise FileNotFoundError(f"shared_seeds.json not found at {SHARED_SEEDS_PATH}")
with open(SHARED_SEEDS_PATH, "r", encoding="utf-8") as f:
    SHARED_SEEDS = json.load(f)
print(f"Loaded {len(SHARED_SEEDS)} shared seeds")
TMP_ZONOS_JSON = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "tmp_directed_zonos_single.json"

HAS_ZONOS = args.condition in ("multimodal", "dissonance")
if HAS_ZONOS:
    SYNTH_DIR = BASE_DIR / "dissonance" / "own_script" / "dialogue_6"
    sys.path.insert(0, str(SYNTH_DIR))
    from run_synthesis_dialogue_6_module import synth_single_utterance

HAS_DELTA = args.condition == "dissonance"
PREFIX = f"{args.condition}_{args.model}"

# Per-condition output directories
DIALOGUE_OUTPUT_DIR = BASE_DIR / "dissonance" / "multibackbone" / args.model / args.condition
DIALOGUE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if HAS_ZONOS:
    VOICE_OUTPUT_DIR = DIALOGUE_OUTPUT_DIR / "voice_output"
    VOICE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import shutil

# Condition labels for output JSON
CONDITION_LABELS = {
    "baseline": "baseline",
    "emotion": "emotion_text_only",
    "multimodal": "multimodal_vocal_aware",
    "dissonance": "dissonance_aware_therapist",
}
OUTPUT_PREFIX = {
    "baseline": "baseline",
    "emotion": "emotion",
    "multimodal": "dissonance",  # matches original batch script naming
    "dissonance": "dissonance",
}

print("=" * 60)
print(f"INFERENCE TIMING: {args.model} / {args.condition}")
print(f"Dialogues: {args.start}-{args.end}")
print("=" * 60)

# =====================================================
# MODEL_MAP
# =====================================================
MODEL_MAP = {
    "gpt4o-mini": "openai/gpt-4o-mini",
    "claude": "anthropic/claude-sonnet-4.6",
    "qwen": "qwen/qwen-2.5-72b-instruct:deepinfra",
    "deepseek": "deepseek/deepseek-chat",
}
MODEL_ID = MODEL_MAP[args.model]

# =====================================================
# OpenRouter client
# =====================================================
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY", ""))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Shared models (load once)
# =====================================================
VAD_MODEL_NAME = "RobroKools/vad-bert"
tokenizer = AutoTokenizer.from_pretrained(VAD_MODEL_NAME)
vad_model = AutoModelForSequenceClassification.from_pretrained(VAD_MODEL_NAME).to(device); vad_model.eval()

if HAS_ZONOS:
    WAVLM_MODEL_NAME = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
    _wavlm = AutoModelForAudioClassification.from_pretrained(WAVLM_MODEL_NAME, trust_remote_code=True).to(device); _wavlm.eval()
    _target_sr = _wavlm.config.sampling_rate
    _mean = _wavlm.config.mean; _std = _wavlm.config.std

# =====================================================
# Helpers (with timing)
# =====================================================
def _to_minus1_1(x, xmin=1.0, xmax=5.0): return float(2*(x-xmin)/(xmax-xmin)-1)

def timed(fn, *a, **kw):
    t0 = time.perf_counter(); result = fn(*a, **kw); elapsed = time.perf_counter() - t0
    if fn is chat_once_robust:
        elapsed = chat_once_robust._api_time  # exclude retry sleep time
    return result, elapsed

# =====================================================
# Helper functions
# =====================================================
def get_text_VA(text: str) -> Tuple[float, float]:
    enc = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad(): out = vad_model(**enc)
    v, a, _ = out.logits.cpu().numpy()[0].tolist()
    return _to_minus1_1(v, 1.0, 5.0), _to_minus1_1(a, 1.0, 5.0)

def get_discrete_emotion(val: float, aro: float) -> str:
    d = math.sqrt(val**2+aro**2)
    if d < 0.15: return "Neutral"
    angle = math.degrees(math.atan2(aro, val))
    if angle < 0: angle += 360
    centers = [(0,"Happy"),(45,"Excited"),(90,"Tense"),(135,"Angry"),(180,"Sad"),(225,"Depressed"),(270,"Calm"),(315,"Content")]
    return min(centers, key=lambda c: min(abs(angle-c[0]), 360-abs(angle-c[0])))[1]

def chat_once_robust(system_prompt, user_prompt):
    """LLM call with retries. Excludes sleep time from API timing via _api_time attr."""
    attempt = 0
    chat_once_robust._api_time = 0.0
    while True:
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                temperature=0.7, max_tokens=512,
            )
            chat_once_robust._api_time += time.perf_counter() - t0
            content = resp.choices[0].message.content
            if content and content.strip(): return content.strip()
            attempt += 1
            print(f"  [RETRY] Empty response, attempt {attempt}, waiting 2s")
            time.sleep(2)
        except Exception as e:
            attempt += 1
            wait = min(2**attempt, 60)
            print(f"  [RETRY] {e}, attempt {attempt}, waiting {wait}s")
            time.sleep(wait)

def get_min_sec_per_char(val_t, aro_t):
    return max(0.020, min(0.060, 0.035 + val_t*-0.002 + aro_t*-0.007))

# =====================================================
# Zonos-side functions (only when needed)
# =====================================================
if HAS_ZONOS:
    def _predict_file(path: str) -> Tuple[float, float, float]:
        audio, sr = sf.read(path)
        if audio.ndim > 1: audio = audio.mean(axis=1)
        if sr != _target_sr: audio = librosa.resample(audio, orig_sr=sr, target_sr=_target_sr)
        audio = (audio - _mean) / (_std + 1e-6)
        wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        mask = torch.ones(1, wavs.shape[1], dtype=torch.float32).to(device)
        with torch.no_grad(): pred = _wavlm(wavs, mask)
        logits = pred.cpu().numpy()[0].astype(float)
        return float(logits[0]), float(logits[1]), float(logits[2])

    def get_speech_VA(wav_path: Path) -> Tuple[float, float]:
        aro, dom, val = _predict_file(str(wav_path))
        return (2.0*val-1.0), (2.0*aro-1.0)

    def get_vocal_descriptors(wav_path: Path, client_text: str) -> str:
        y, sr = librosa.load(str(wav_path), sr=None)
        dur = len(y)/sr
        pitches, _ = librosa.piptrack(y=y, sr=sr); pv = pitches[pitches>0]
        pm = float(pv.mean()) if len(pv) else 0
        rms = float(librosa.feature.rms(y=y).mean())
        rate = len(client_text.split())/dur if dur else 1.0
        pd = "very low pitch" if pm<100 else "low pitch" if pm<150 else "moderate pitch" if pm<200 else "high pitch" if pm<250 else "very high pitch"
        ld = "very quiet" if rms<0.02 else "soft-spoken" if rms<0.05 else "moderate volume" if rms<0.10 else "loud" if rms<0.15 else "very loud"
        rd = "slow speech" if rate<2.0 else "moderate-paced speech" if rate<3.0 else "fast speech" if rate<4.0 else "very rapid speech"
        return f"{pd}, {ld}, {rd}"

    ZONOS_DIRECTOR_SYSTEM = """You are a master Vocal Director. Given ONE client utterance from a CBT therapy session, output a JSON object with a single directed utterance for Zonos. Schema: {"utterance_text":"...","is_new_utterance_rule":true,"utterance_level_direction":"...","new_utterance_rule_definition":{"primary_zonos_vector_value":{"Happiness":0,"Sadness":0.8,"Fear":0.4},"speaking_rate":15.0,"pitch_std":100.0},"frame_level_directions":[]}. Copy utterance_text EXACTLY. Values -1.0 to 1.0. speaking_rate 10-25. pitch_std 20-150. Output ONLY JSON."""

    def make_directed_zonos_for_text(client_text: str) -> dict:
        raw = chat_once_robust(ZONOS_DIRECTOR_SYSTEM, f'Client utterance:\n\n"""\n{client_text}\n"""\n\nOutput ONLY JSON.')
        import re
        raw = raw.replace("```json","").replace("```","")
        start = raw.find("{"); end = raw.rfind("}")+1
        json_str = raw[start:end] if start>=0 and end>start else raw
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        return json.loads(json_str)

    def write_tmp_zonos_json(directed: dict):
        TMP_ZONOS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with TMP_ZONOS_JSON.open("w", encoding="utf-8") as f:
            json.dump({"directed_utterances":[directed]}, f, ensure_ascii=False, indent=2)

    def synthesize_client_audio(client_text: str, turn: int, dialogue_id: int, min_sec_per_char: float = None) -> Path:
        directed = make_directed_zonos_for_text(client_text)
        write_tmp_zonos_json(directed)
        out = synth_single_utterance(turn, str(TMP_ZONOS_JSON), dialogue_id=dialogue_id, prefix=PREFIX, min_sec_per_char=min_sec_per_char)
        if out is None: raise RuntimeError(f"Zonos failed utterance {turn}")
        wav = Path(out)
        if not wav.exists(): raise FileNotFoundError(str(wav))
        return wav

# =====================================================
# Client prompts
# =====================================================
CLIENT_SYSTEM = """You are a CBT therapy client talking to therapist "Luna". You struggle with anxiety, guilt, and loneliness. You may partially resist, question advice, or bring up obstacles. Speak naturally in first-person, 2-3 sentences per turn. Focus on ONE life problem per dialogue. Do NOT use asterisks or stage directions."""
CLIENT_USER_TEMPLATE_FIRST = """Start the first message. Describe what has been bothering you (2-3 sentences).\n\nLife problem scenario:\n{problem_seed}"""
CLIENT_USER_TEMPLATE_NEXT = """Therapist: "{therapist_text}"\nContinue as client. 2-3 sentences. You may express doubts."""

# =====================================================
# Therapist prompts
# =====================================================
if args.condition == "baseline":
    THERAPIST_SYSTEM = """You are "Luna", a CBT therapist. Receive only dialogue transcript. Respond with empathy, 2-3 sentences. Do not mention analysis."""
    THERAPIST_TEMPLATE = """Client: "{client_text}"\nRespond."""
elif args.condition == "emotion":
    THERAPIST_SYSTEM = """You are "Luna", a CBT therapist. You receive text VA values and text emotion. Use them to guide empathy and CBT techniques. 2-3 sentences. Do not mention numbers."""
    THERAPIST_TEMPLATE = """Client: "{client_text}"\nText VA: val={val_t:.2f}, aro={aro_t:.2f} | Emo: {text_emotion}\nRespond."""
elif args.condition == "multimodal":
    THERAPIST_SYSTEM = """You are "Luna", a CBT therapist. You receive text VA, speech VA, and vocal descriptors. Use both channels to guide empathy. 2-3 sentences. No numbers."""
    THERAPIST_TEMPLATE = """Client: "{client_text}"\nText VA: val={val_t:.2f}, aro={aro_t:.2f}\nSpeech VA: val={val_s:.2f}, aro={aro_s:.2f}\nVocal: {vocal_descriptors}\nEmo: {text_emotion}/{speech_emotion}\nRespond."""
elif args.condition == "dissonance":
    THERAPIST_SYSTEM = """You are "Luna", a CBT therapist. You receive text VA, speech VA, vocal descriptors, and dissonance (delta + mismatch flag). When dissonance is high, gently explore hidden affect. 2-3 sentences. No numbers."""
    THERAPIST_TEMPLATE = """Client: "{client_text}"\nText VA: val={val_t:.2f}, aro={aro_t:.2f}\nSpeech VA: val={val_s:.2f}, aro={aro_s:.2f}\nVocal: {vocal_descriptors}\nEmo: {text_emotion}/{speech_emotion}\nDelta: dv={delta_v:.2f} da={delta_a:.2f} | Dissonance: {is_dissonant} | Mismatch: {emotion_mismatch}\nRespond."""

# =====================================================
# Timing per turn
# =====================================================
TurnTimings = Dict[str, float]

def run_single_dialogue(dialogue_id: int, max_turns: int = 10) -> List[TurnTimings]:
    seed = SHARED_SEEDS[dialogue_id-1]
    print(f"\nDialogue {dialogue_id}")
    all_timings = []
    turns = []  # dialogue records for JSONL output

    # --- Turn 1 ---
    t = {"turn": 1}

    client_text, t["client_llm"] = timed(chat_once_robust, CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_FIRST.format(problem_seed=seed))
    print(f"  Client: {client_text[:60]}... ({t['client_llm']:.2f}s)")

    (val_t, aro_t), t["vadbert"] = timed(get_text_VA, client_text)
    d_emo, t["discrete_text"] = timed(get_discrete_emotion, val_t, aro_t)
    rate = get_min_sec_per_char(val_t, aro_t)

    turn_record = {
        "turn": 1, "client": client_text, "therapist": "",
        "condition": CONDITION_LABELS[args.condition], "model": args.model,
        "val_t": val_t, "aro_t": aro_t,
    }
    if not HAS_ZONOS:
        turn_record["text_emotion"] = d_emo

    if HAS_ZONOS:
        _, t["director_llm"] = timed(make_directed_zonos_for_text, client_text)
        wav_path = synthesize_client_audio(client_text, turn=1, dialogue_id=dialogue_id, min_sec_per_char=rate)
        # Copy WAV to per-condition voice directory
        if HAS_ZONOS:
            voice_dst = VOICE_OUTPUT_DIR / wav_path.name
            shutil.copy2(str(wav_path), str(voice_dst))
            wav_path = voice_dst
        (val_s, aro_s), t["wavlm"] = timed(get_speech_VA, wav_path)
        voc, t["librosa"] = timed(get_vocal_descriptors, wav_path, client_text)
        d_speech, t["discrete_speech"] = timed(get_discrete_emotion, val_s, aro_s)
        turn_record.update({"val_s": val_s, "aro_s": aro_s, "text_emotion": d_emo, "speech_emotion": d_speech,
                            "vocal_descriptors": voc, "audio_path": str(wav_path)})

    if HAS_DELTA:
        dv = val_s - val_t; da = aro_s - aro_t
        t0d = time.perf_counter()
        is_dis = (abs(dv) >= 0.5) or (abs(da) >= 0.5)
        emo_mm = (d_emo != d_speech)
        t["dissonance_comp"] = time.perf_counter() - t0d
        turn_record.update({"delta_valence": dv, "delta_arousal": da,
                            "is_dissonant": is_dis, "emotion_mismatch": emo_mm})

    if args.condition == "baseline":
        therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text))
    elif args.condition == "emotion":
        therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text, val_t=val_t, aro_t=aro_t, text_emotion=d_emo))
    elif args.condition == "multimodal":
        therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text, val_t=val_t, aro_t=aro_t, val_s=val_s, aro_s=aro_s, vocal_descriptors=voc, text_emotion=d_emo, speech_emotion=d_speech))
    elif args.condition == "dissonance":
        therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text, val_t=val_t, aro_t=aro_t, val_s=val_s, aro_s=aro_s, vocal_descriptors=voc, text_emotion=d_emo, speech_emotion=d_speech, delta_v=dv, delta_a=da, is_dissonant=is_dis, emotion_mismatch=emo_mm))

    turn_record["therapist"] = therapist_text
    turns.append(turn_record)
    print(f"  Therapist: {therapist_text[:60]}... ({t['therapist_llm']:.2f}s)")
    all_timings.append(t)

    # --- Turns 2-max_turns ---
    for turn_idx in range(2, max_turns + 1):
        t = {"turn": turn_idx}

        client_text, t["client_llm"] = timed(chat_once_robust, CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_NEXT.format(therapist_text=therapist_text))
        print(f"  Client (t={turn_idx}): {client_text[:60]}... ({t['client_llm']:.2f}s)")
        (val_t, aro_t), t["vadbert"] = timed(get_text_VA, client_text)
        d_emo, t["discrete_text"] = timed(get_discrete_emotion, val_t, aro_t)
        rate = get_min_sec_per_char(val_t, aro_t)

        turn_record = {
            "turn": turn_idx, "client": client_text, "therapist": "",
            "condition": CONDITION_LABELS[args.condition], "model": args.model,
            "val_t": val_t, "aro_t": aro_t,
        }
        if not HAS_ZONOS:
            turn_record["text_emotion"] = d_emo

        if HAS_ZONOS:
            _, t["director_llm"] = timed(make_directed_zonos_for_text, client_text)
            wav_path = synthesize_client_audio(client_text, turn=turn_idx, dialogue_id=dialogue_id, min_sec_per_char=rate)
            voice_dst = VOICE_OUTPUT_DIR / wav_path.name
            shutil.copy2(str(wav_path), str(voice_dst))
            wav_path = voice_dst
            (val_s, aro_s), t["wavlm"] = timed(get_speech_VA, wav_path)
            voc, t["librosa"] = timed(get_vocal_descriptors, wav_path, client_text)
            d_speech, t["discrete_speech"] = timed(get_discrete_emotion, val_s, aro_s)
            turn_record.update({"val_s": val_s, "aro_s": aro_s, "text_emotion": d_emo, "speech_emotion": d_speech,
                                "vocal_descriptors": voc, "audio_path": str(wav_path)})

        if HAS_DELTA:
            dv = val_s - val_t; da = aro_s - aro_t
            t0d = time.perf_counter()
            is_dis = (abs(dv) >= 0.5) or (abs(da) >= 0.5)
            emo_mm = (d_emo != d_speech)
            t["dissonance_comp"] = time.perf_counter() - t0d
            turn_record.update({"delta_valence": dv, "delta_arousal": da,
                                "is_dissonant": is_dis, "emotion_mismatch": emo_mm})

        if args.condition == "baseline":
            therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text))
        elif args.condition == "emotion":
            therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text, val_t=val_t, aro_t=aro_t, text_emotion=d_emo))
        elif args.condition == "multimodal":
            therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text, val_t=val_t, aro_t=aro_t, val_s=val_s, aro_s=aro_s, vocal_descriptors=voc, text_emotion=d_emo, speech_emotion=d_speech))
        elif args.condition == "dissonance":
            therapist_text, t["therapist_llm"] = timed(chat_once_robust, THERAPIST_SYSTEM, THERAPIST_TEMPLATE.format(client_text=client_text, val_t=val_t, aro_t=aro_t, val_s=val_s, aro_s=aro_s, vocal_descriptors=voc, text_emotion=d_emo, speech_emotion=d_speech, delta_v=dv, delta_a=da, is_dissonant=is_dis, emotion_mismatch=emo_mm))

        turn_record["therapist"] = therapist_text
        print(f"  Therapist (t={turn_idx}): {therapist_text[:60]}... ({t['therapist_llm']:.2f}s)")
        turns.append(turn_record)
        all_timings.append(t)

    # Save dialogue JSONL
    FILE_PREFIX = {"baseline": "baseline", "emotion": "emotion", "multimodal": "dissonance", "dissonance": "dissonance"}
    FILE_SUFFIX = {"baseline": "baseline", "emotion": "emotion_online", "multimodal": "multimodal", "dissonance": "dissonance_online"}
    jsonl_name = f"{FILE_PREFIX[args.condition]}_{dialogue_id}_full_{FILE_SUFFIX[args.condition]}"
    jsonl_path = DIALOGUE_OUTPUT_DIR / f"{jsonl_name}.jsonl"
    json_path = DIALOGUE_OUTPUT_DIR / f"{jsonl_name}.json"
    for p in [jsonl_path, json_path]:
        with open(p, "w", encoding="utf-8") as f:
            if p.suffix == ".json": json.dump(turns, f, ensure_ascii=False, indent=2)
            else:
                for r in turns: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [SAVED] {jsonl_path} ({len(turns)} turns)")

    return all_timings

# =====================================================
# Per-dialogue JSON save + Markdown report
# =====================================================
TIMING_DIR = INFERENCE_DIR / f"{args.model}_{args.condition}"
TIMING_DIR.mkdir(parents=True, exist_ok=True)

def save_dialogue_timing(dialogue_id: int, timings: List[TurnTimings]):
    path = TIMING_DIR / f"dialogue_{dialogue_id}_infertime.json"
    dtotal = sum(
        v for t in timings for k, v in t.items()
        if isinstance(v, (int, float)) and k != "turn"
    )
    data = {"dialogue_id": dialogue_id, "total_seconds": dtotal, "turns": timings}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  [SAVED] {path}")

def write_report(model_name: str, condition: str):
    md_path = INFERENCE_DIR / f"{model_name}_{condition}_infertime.md"
    json_files = sorted(TIMING_DIR.glob("dialogue_*_infertime.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not json_files:
        print("No timing data found.")
        return

    # Load all
    all_data = [json.load(open(p, "r")) for p in json_files]
    all_turns = [turn for d in all_data for turn in d["turns"]]

    # Components
    components = set()
    for d in all_data:
        for t in d["turns"]:
            components.update(k for k in t if k not in ("turn", "text_emotion", "speech_emotion", "vocal_desc"))
    components = sorted(components)

    agg = {c: [] for c in components}
    dia_totals = []

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Inference Timing: {model_name} / {condition}\n\n")
        f.write(f"**Model:** `{MODEL_ID}` | **Condition:** {condition} | **Dialogues:** {len(all_data)} | **Turns:** {len(all_turns)}\n\n")
        f.write("---\n\n")

        for d in all_data:
            did = d["dialogue_id"]
            dia_totals.append(d["total_seconds"])
            f.write(f"## Dialogue {did}\n\n")
            header = "| Turn |" + "".join(f" {c.replace('_',' ').title()} |" for c in components) + " **Total** |\n"
            f.write(header)
            f.write("|" + "---|" * (len(components) + 2) + "\n")

            for t in d["turns"]:
                row = f"| {t['turn']} |"
                tsum = 0
                for c in components:
                    val = t.get(c)
                    if isinstance(val, (int, float)):
                        row += f" {val:.4f}s |"
                        tsum += val
                        agg[c].append(val)
                    else:
                        row += " — |"
                row += f" **{tsum:.2f}s** |"
                f.write(row + "\n")
            f.write(f"\n**Dialogue {did} total (excl Zonos): {d['total_seconds']:.1f}s**\n\n")

        # Aggregate
        f.write("---\n\n## Aggregate\n\n")
        f.write("| Component | Count | Avg | StdDev | Min | Max | Total |\n")
        f.write("|" + "---|" * 7 + "\n")
        for c in components:
            vals = agg[c]
            if not vals: continue
            f.write(f"| {c.replace('_',' ').title()} | {len(vals)} | {statistics.mean(vals):.4f}s | {statistics.stdev(vals):.4f}s | {min(vals):.4f}s | {max(vals):.4f}s | {sum(vals):.2f}s |\n")

        total_all = sum(dia_totals)
        f.write(f"\n**Total (all dialogues, excl Zonos): {total_all:.1f}s**\n")
        f.write(f"**Avg per dialogue: {statistics.mean(dia_totals):.1f}s**\n")

    print(f"\nSaved: {md_path} ({len(all_data)} dialogues)")

# =====================================================
# Main
# =====================================================
for did in range(args.start, args.end + 1):
    try:
        timings = run_single_dialogue(did)
        save_dialogue_timing(did, timings)
    except Exception as e:
        print(f"ERROR dialogue {did}: {e}")

# Always regenerate report from all saved JSONs
write_report(args.model, args.condition)
