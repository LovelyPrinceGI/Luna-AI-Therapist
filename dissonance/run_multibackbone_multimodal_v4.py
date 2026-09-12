#!/usr/bin/env python3
"""
Multi-backbone Dissonance-Aware pipeline via OpenRouter.
Re-runs full pipeline (client + Zonos Director + therapist) with different LLM backbones.
Each backbone gets: per-model output dir + unique WAV filenames + unique audio_paths.

Usage:
    python run_dissonance_multibackbone.py --model claude --start 1 --end 100
    python run_dissonance_multibackbone.py --model gemini --start 1 --end 100
    python run_dissonance_multibackbone.py --model deepseek --start 1 --end 100
"""
import argparse
import os
import sys
import json
import math
import re
import gc
import time
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import soundfile as sf
import librosa
import random
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForAudioClassification

# =====================================================
# Platform-aware paths
# =====================================================
import platform
if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

SYNTH_DIR = BASE_DIR / "dissonance" / "own_script" / "dialogue_6"
sys.path.insert(0, str(SYNTH_DIR))
from run_synthesis_dialogue_6_module import synth_single_utterance

OUTPUT_BASE = BASE_DIR / "dissonance" / "multibackbone"
VOICE_BASE = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "voice"
VOICE_BASE.mkdir(parents=True, exist_ok=True)

TMP_ZONOS_JSON = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "tmp_directed_zonos_single.json"

# =====================================================
# OpenRouter model map
# =====================================================
MODEL_MAP = {
    "gpt4o-mini": "openai/gpt-4o-mini",
    "claude": "anthropic/claude-sonnet-4.6",
    "qwen": "qwen/qwen-2.5-72b-instruct:deepinfra",
    "deepseek": "deepseek/deepseek-chat",
}

# =====================================================
# CLI
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()),
                    help="LLM backbone to use")
parser.add_argument("--start", type=int, default=1)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--max-turns", type=int, default=10)
parser.add_argument("--manip-rate", type=float, default=0.0,
                    help="Controlled masked-affect manipulation rate (0 = original method)")
args = parser.parse_args()
MANIP_RATE = args.manip_rate

MODEL_NAME = args.model
MODEL_ID = MODEL_MAP[MODEL_NAME]
OUTPUT_DIR = OUTPUT_BASE / MODEL_NAME / "multimodal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREFIX = f"multimodal_{MODEL_NAME}"

print("=" * 60)
print(f"MULTI-BACKBONE DISSONANCE: {MODEL_NAME} ({MODEL_ID})")
print(f"Output dir: {OUTPUT_DIR}")
print(f"WAV prefix: {PREFIX}")
print(f"Dialogues: {args.start}-{args.end}")
print("=" * 60)

# =====================================================
# OpenRouter client
# =====================================================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================================================
# Text VA: vad-bert
# =====================================================
VAD_MODEL_NAME = "RobroKools/vad-bert"
tokenizer = AutoTokenizer.from_pretrained(VAD_MODEL_NAME)
vad_model = AutoModelForSequenceClassification.from_pretrained(VAD_MODEL_NAME).to(device)
vad_model.eval()

V_MIN, V_MAX = 1.0, 5.0
A_MIN, A_MAX = 1.0, 5.0

def _to_minus1_1(x, xmin=1.0, xmax=5.0): return float(2 * (x - xmin) / (xmax - xmin) - 1.0)

def get_text_VA(text: str) -> Tuple[float, float]:
    enc = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = vad_model(**enc)
    v_raw, a_raw, d_raw = out.logits.cpu().numpy()[0].tolist()
    return _to_minus1_1(v_raw, V_MIN, V_MAX), _to_minus1_1(a_raw, A_MIN, A_MAX)

# =====================================================
# WavLM: speech VA
# =====================================================
WAVLM_MODEL_NAME = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
print(f"Loading WavLM: {WAVLM_MODEL_NAME} ...")
_wavlm = AutoModelForAudioClassification.from_pretrained(WAVLM_MODEL_NAME, trust_remote_code=True).to(device)
_wavlm.eval()
_target_sr = _wavlm.config.sampling_rate
_mean = _wavlm.config.mean
_std = _wavlm.config.std

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
    return (2.0 * val - 1.0), (2.0 * aro - 1.0)

# =====================================================
# Vocal Descriptors: librosa
# =====================================================
def get_vocal_descriptors(wav_path: Path, client_text: str) -> str:
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

# =====================================================
# Discrete Emotion: Russell (1980) circumplex
# =====================================================
def get_discrete_emotion(val: float, aro: float) -> str:
    distance = math.sqrt(val**2 + aro**2)
    if distance < 0.15: return "Neutral"
    angle = math.degrees(math.atan2(aro, val))
    if angle < 0: angle += 360
    centers = [(0, "Happy"), (45, "Excited"), (90, "Tense"), (135, "Angry"),
               (180, "Sad"), (225, "Depressed"), (270, "Calm"), (315, "Content")]
    closest = min(centers, key=lambda c: min(abs(angle - c[0]), 360 - abs(angle - c[0])))
    return closest[1]

# Dynamic speaking rate
def get_min_sec_per_char(val_t: float, aro_t: float) -> float:
    base = 0.035
    va_factor = val_t * -0.002 + aro_t * -0.007
    return max(0.020, min(0.060, base + va_factor))

# =====================================================
# OpenRouter LLM helper
# =====================================================
def chat_once(system_prompt: str, user_prompt: str, history: list = None) -> str:
    import time
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0.7, max_tokens=512,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
            attempt += 1
            print(f"  [RETRY] Empty response, attempt {attempt}")
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt, 60)
            print(f"  [RETRY] {e}, attempt {attempt}, waiting {wait}s")
            time.sleep(wait)
            continue
        time.sleep(1)

# =====================================================
# Zonos Director LLM
# =====================================================
ZONOS_DIRECTOR_SYSTEM = """
You are a master Vocal Director simulating the emotion2vec framework.

Given ONE client utterance from a CBT therapy session, you must output
a JSON object with a single directed utterance for Zonos, matching this schema:

{
  "utterance_text": "...",
  "is_new_utterance_rule": true,
  "utterance_level_direction": "[anxious, slow]",
  "new_utterance_rule_definition": {
    "primary_zonos_vector_value": {
      "Happiness": 0.0, "Sadness": 0.8, "Fear": 0.4
    },
    "speaking_rate": 15.0,
    "pitch_std": 100.0
  },
  "frame_level_directions": []
}

Rules:
- Copy the client utterance EXACTLY into "utterance_text".
- Use only these emotion keys: Happiness, Sadness, Disgust, Fear, Surprise, Anger, Neutral, Other.
- Values between -1.0 and 1.0.
- speaking_rate: 10.0 to 25.0
- pitch_std: 20.0 to 150.0
- is_new_utterance_rule must always be true.
- Output ONLY the JSON object. No extra commentary.
"""

DIRECTOR_EMOTION_KEYS = ["Happiness", "Sadness", "Disgust", "Fear", "Surprise", "Anger", "Other", "Neutral"]

def _repair_director_json(raw: str):
    """Best-effort parse of director JSON. Handles truncation, unquoted keys,
    trailing commas, and missing emotion keys. Returns dict or None."""
    s = raw.replace("```json", "").replace("```", "").strip()
    start = s.find("{")
    if start < 0:
        return None
    s = s[start:]
    end = s.rfind("}")
    if end >= 0:
        s = s[:end + 1]
    else:
        # Truncated output: strip dangling tail, then close open brackets
        s = re.sub(r',\s*"[^"]*"\s*:\s*[^,}\]]*$', '', s)
        s = re.sub(r',\s*"[^"]*$', '', s)
        s = re.sub(r',\s*$', '', s)
        s = s + ']' * max(s.count('[') - s.count(']'), 0) + '}' * max(s.count('{') - s.count('}'), 0)
    # Fix common LLM JSON errors: unquoted keys + trailing commas
    s = re.sub(r'([{,]\s*)([a-zA-Z_]\w*)\s*:', r'\1"\2":', s)
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    rule = d.get("new_utterance_rule_definition")
    if not isinstance(rule, dict):
        rule = {}
    vec = rule.get("primary_zonos_vector_value")
    if not isinstance(vec, dict) or not vec:
        vec = {"Neutral": 0.8}
    rule["primary_zonos_vector_value"] = {k: float(vec.get(k, 0.0)) for k in DIRECTOR_EMOTION_KEYS}
    rule.setdefault("speaking_rate", 15.0)
    rule.setdefault("pitch_std", 60.0)
    d["new_utterance_rule_definition"] = rule
    d.setdefault("is_new_utterance_rule", True)
    d.setdefault("utterance_level_direction", "neutral")
    d.setdefault("frame_level_directions", [])
    return d

def _neutral_director(client_text: str) -> dict:
    return {
        "utterance_text": client_text,
        "is_new_utterance_rule": True,
        "utterance_level_direction": "neutral",
        "new_utterance_rule_definition": {
            "primary_zonos_vector_value": {k: (0.8 if k == "Neutral" else 0.0) for k in DIRECTOR_EMOTION_KEYS},
            "speaking_rate": 15.0,
            "pitch_std": 60.0,
        },
        "frame_level_directions": [],
    }

def make_directed_zonos_for_text(client_text: str) -> dict:
    prompt = (f'Client utterance:\n\n"""\n{client_text}\n"""\n\n'
              f'Generate ONE directed utterance JSON. Output only the JSON.')
    raw = chat_once(ZONOS_DIRECTOR_SYSTEM, prompt)
    for attempt in range(3):
        d = _repair_director_json(raw)
        if d is not None:
            d["utterance_text"] = client_text  # ensure exact text
            return d
        print(f"  [DIRECTOR-RETRY {attempt + 1}/3] malformed JSON, asking again...")
        raw = chat_once(ZONOS_DIRECTOR_SYSTEM,
                        prompt + "\n\nYour previous output was malformed or truncated. Output the COMPLETE JSON object again, with no commentary.")
    print("  [DIRECTOR-FALLBACK] using neutral director after repeated parse failures")
    return _neutral_director(client_text)

def write_tmp_zonos_json(directed: dict):
    TMP_ZONOS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with TMP_ZONOS_JSON.open("w", encoding="utf-8") as f:
        json.dump({"directed_utterances": [directed]}, f, ensure_ascii=False, indent=2)

def contrast_director(val_t: float) -> dict:
    if val_t < 0:
        return {
            "primary_zonos_vector_value": {
                "Happiness": 0.9, "Sadness": -0.7, "Fear": -0.4, "Anger": -0.4,
                "Surprise": 0.2, "Disgust": -0.3, "Neutral": -0.6, "Other": 0.0
            },
            "speaking_rate": 22.0,
            "pitch_std": 130.0
        }
    return {
        "primary_zonos_vector_value": {
            "Happiness": -0.7, "Sadness": 0.9, "Fear": 0.7, "Anger": 0.5,
            "Surprise": -0.3, "Disgust": -0.2, "Neutral": -0.6, "Other": 0.0
        },
        "speaking_rate": 12.0,
        "pitch_std": 40.0
    }

def synthesize_client_audio(client_text: str, turn: int, dialogue_id: int, min_sec_per_char: float = None, val_t: float = None) -> Path:
    directed = make_directed_zonos_for_text(client_text)
    if MANIP_RATE > 0 and val_t is not None:
        rng = random.Random(f"{dialogue_id}-{turn}")
        if rng.random() < MANIP_RATE:
            directed["new_utterance_rule_definition"] = contrast_director(val_t)
            directed["utterance_level_direction"] = "cheerful, energetic" if val_t < 0 else "tense, flat"
            print(f"  [MANIP] turn {turn}: contrast voice injected (val_t={val_t:+.2f})")
    write_tmp_zonos_json(directed)
    print(f"  [TURN {turn}] Zonos synth (rate={min_sec_per_char}) ...")
    out_path_str = synth_single_utterance(turn, str(TMP_ZONOS_JSON), dialogue_id=dialogue_id, prefix=PREFIX, min_sec_per_char=min_sec_per_char)
    if out_path_str is None: raise RuntimeError(f"Zonos failed to synthesize utterance {turn}")
    wav_path = Path(out_path_str)
    if not wav_path.exists(): raise FileNotFoundError(f"Expected audio not found: {wav_path}")
    return wav_path

# =====================================================
# Client prompts (resist persona)
# =====================================================
CLIENT_SYSTEM = """
You are a CBT therapy client talking to therapist "Luna".

- You struggle with anxiety, guilt, and loneliness in your life.
- You sometimes feel misunderstood or skeptical about therapy.
- When the therapist suggests reframing, advice, or homework,
  you may partially resist, question it, or bring up obstacles
  (e.g., "I don't think that will work for me", "It's hard because ...").
- Speak in a natural, first-person voice.
- Stay emotionally consistent across turns.
- Describe thoughts, feelings, and situations in 2–3 sentences per turn.
- Do NOT use asterisks (*), stage directions, or role-play actions.
  Speak only in plain text as a real person would.
- In each full dialogue, you must focus on only ONE life problem scenario.
- Do not mix multiple problem seeds in the same dialogue.
- Once a problem seed is assigned for a dialogue, keep that same core life problem throughout the whole dialogue.
"""

CLIENT_USER_TEMPLATE_FIRST = """
Start the first message to your therapist.

Describe what has been bothering you lately (2–3 sentences).
You may already feel unsure whether therapy can really help.

Important:
- This dialogue has exactly ONE assigned life problem scenario.
- You must only use the following scenario in this whole dialogue.
- Do not introduce a second major life problem.

Assigned life problem scenario:
{problem_seed}
"""

CLIENT_USER_TEMPLATE_NEXT = """
Therapist just said:
"{therapist_text}"

Continue the conversation as the client.
Describe what you think and feel now in 2–3 sentences.
If the therapist gives advice, interpretations, or homework,
you can question it, express doubts, or explain why it feels difficult.

Important:
- Stay within the same assigned life problem scenario for this whole dialogue.
- Do not switch to a new major life problem.
"""

# =====================================================
# Shared seeds
# =====================================================
SHARED_SEEDS_PATH = OUTPUT_BASE / "shared_seeds.json"
if not SHARED_SEEDS_PATH.exists():
    raise FileNotFoundError(f"Run generate_shared_seeds.py first!\nExpected: {SHARED_SEEDS_PATH}")
with open(SHARED_SEEDS_PATH, "r", encoding="utf-8") as f:
    SHARED_SEEDS = json.load(f)
print(f"Loaded {len(SHARED_SEEDS)} shared seeds")

# =====================================================
# Therapist prompts (Dissonance-Aware)
# =====================================================
THERAPIST_SYSTEM = """
You are "Luna", a CBT therapist with access to both the client's words and
an analysis of their voice emotion.

For each client message you receive:
- Text-based emotion: Valence_text, Arousal_text (from -1 to +1)
- Voice-based emotion: Valence_speech, Arousal_speech (from -1 to +1)
- Vocal prosodic descriptors: pitch level, loudness, speaking rate

Interpretation guidelines:
- Valence reflects how positive or negative the emotion is.
- Arousal reflects the intensity or activation level.
- Use both text and voice emotional signals together to better understand
  the client's overall cognitive and affective state.

Your job:
- Respond with empathy, using CBT principles (thoughts, evidence,
  alternative perspectives).
- Use the dual emotional signals (text + voice) to guide the depth
  of your reflection and response.
- Reflect what the client may be feeling, informed by both their
  words and their tone.

Important:
- NEVER mention numbers, "VA", "valence", "arousal", or "analysis".
- Speak only in natural language.
- Reply in 2–3 sentences.
- Still follow CBT principles.
"""

THERAPIST_USER_TEMPLATE = """
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

Write your next therapist response.
Use both text and vocal emotional cues to guide the depth of your empathy.
Do NOT mention any numbers or analysis terms in your response.
"""

# =====================================================
# Save helper
# =====================================================
def save_dialogue_json_and_jsonl(turns, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".json", ".jsonl"]:
        p = base_path.with_suffix(suffix)
        with p.open("w", encoding="utf-8") as f:
            if suffix == ".json": json.dump(turns, f, ensure_ascii=False, indent=2)
            else:
                for rec in turns: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [SAVE] {base_path.with_suffix('.jsonl')}")

# =====================================================
# Main loop
# =====================================================
def run_single_dialogue(dialogue_id: int, max_turns: int = 10):
    seed = SHARED_SEEDS[dialogue_id - 1]
    print(f"\n=== {MODEL_NAME.upper()} DISSONANCE dialogue {dialogue_id} ===")

    turns = []
    conversation_history = []

    # Turn 1
    client_text = chat_once(CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_FIRST.format(problem_seed=seed))
    print(f"CLIENT (t=1): {client_text[:80]}...")

    val_t, aro_t = get_text_VA(client_text)
    text_emo = get_discrete_emotion(val_t, aro_t)
    rate = get_min_sec_per_char(val_t, aro_t)
    wav_path = synthesize_client_audio(client_text, turn=1, dialogue_id=dialogue_id, min_sec_per_char=rate, val_t=val_t)
    val_s, aro_s = get_speech_VA(wav_path)
    vocal_desc = get_vocal_descriptors(wav_path, client_text)
    speech_emo = get_discrete_emotion(val_s, aro_s)

    

    print(f"  VA: text=({val_t:.2f},{aro_t:.2f}) speech=({val_s:.2f},{aro_s:.2f}) | emo={text_emo}/{speech_emo}")

    therapist_text = chat_once(THERAPIST_SYSTEM, THERAPIST_USER_TEMPLATE.format(
        client_text=client_text, val_t=val_t, aro_t=aro_t, val_s=val_s, aro_s=aro_s,
        vocal_descriptors=vocal_desc), history=list(conversation_history))
    print(f"THERAPIST (t=1): {therapist_text[:80]}...")
    conversation_history.append({"role": "user", "content": client_text})
    conversation_history.append({"role": "assistant", "content": therapist_text})

    turns.append({
        "turn": 1, "client": client_text, "therapist": therapist_text,
        "condition": "multimodal_vocal_aware", "model": MODEL_NAME,
        "val_t": val_t, "aro_t": aro_t, "val_s": val_s, "aro_s": aro_s,
        "text_emotion": text_emo, "speech_emotion": speech_emo,
        "vocal_descriptors": vocal_desc,
        "audio_path": str(wav_path),
    })

    # Turns 2-10
    for t in range(2, max_turns + 1):
        # v4: client remembers the whole dialogue (roles flipped: client's own
        # past lines are "assistant", therapist's lines are "user")
        client_history = [
            {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
            for m in conversation_history
        ]
        client_text = chat_once(CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_NEXT.format(therapist_text=therapist_text),
                                history=client_history)
        print(f"CLIENT (t={t}): {client_text[:80]}...")

        val_t, aro_t = get_text_VA(client_text)
        text_emo = get_discrete_emotion(val_t, aro_t)
        rate = get_min_sec_per_char(val_t, aro_t)
        wav_path = synthesize_client_audio(client_text, turn=t, dialogue_id=dialogue_id, min_sec_per_char=rate, val_t=val_t)
        val_s, aro_s = get_speech_VA(wav_path)
        vocal_desc = get_vocal_descriptors(wav_path, client_text)
        speech_emo = get_discrete_emotion(val_s, aro_s)



        therapist_text = chat_once(THERAPIST_SYSTEM, THERAPIST_USER_TEMPLATE.format(
            client_text=client_text, val_t=val_t, aro_t=aro_t, val_s=val_s, aro_s=aro_s,
            vocal_descriptors=vocal_desc), history=list(conversation_history))
        print(f"THERAPIST (t={t}): {therapist_text[:80]}...")
        conversation_history.append({"role": "user", "content": client_text})
        conversation_history.append({"role": "assistant", "content": therapist_text})

        turns.append({
            "turn": t, "client": client_text, "therapist": therapist_text,
            "condition": "multimodal_vocal_aware", "model": MODEL_NAME,
            "val_t": val_t, "aro_t": aro_t, "val_s": val_s, "aro_s": aro_s,
            "text_emotion": text_emo, "speech_emotion": speech_emo,
            "vocal_descriptors": vocal_desc,
            "audio_path": str(wav_path),
        })

    base_name = f"dissonance_{dialogue_id}_full_multimodal"
    base_path = OUTPUT_DIR / base_name
    save_dialogue_json_and_jsonl(turns, base_path)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for i in range(args.start, args.end + 1):
        max_retries = 3
        for retry in range(max_retries + 1):
            try:
                run_single_dialogue(dialogue_id=i, max_turns=args.max_turns)
                break  # Success, move to next dialogue
            except Exception as e:
                gc.collect()
                torch.cuda.empty_cache()
                is_oom = "out of memory" in str(e).lower()
                if retry < max_retries:
                    wait = 15 if is_oom else 5
                    print(f"ERROR on dialogue {i} (retry {retry + 1}/{max_retries}): {e}")
                    print(f"  Retrying dialogue {i} in {wait} seconds...")
                    time.sleep(wait)
                else:
                    print(f"FATAL: Dialogue {i} failed after {max_retries} retries: {e}")
                    raise  # Never skip: stop here so the run can be resumed from this dialogue

    print(f"\nDONE. Dialogues {args.start}-{args.end} complete for {MODEL_NAME}.")
    print(f"Output: {OUTPUT_DIR}")
