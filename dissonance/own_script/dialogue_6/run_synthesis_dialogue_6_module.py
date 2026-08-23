#!/usr/bin/env python
"""
Run Zonos synthesis for a SINGLE utterance index of dialogue_6.

ใช้ร่วมกับ dialogue_6_dissonance.py:
- dialogue_6_dissonance จะจัดการสร้าง/อัปเดต JSON ที่มี directed_utterances
- สคริปต์นี้รับ utterance idx (= dialogue turn) แล้ว gen เสียงไปเก็บใน voice/
"""

import argparse
import os
import json
import re
import sys
from typing import Optional

import torch
import torchaudio
import numpy as np
from scipy.io.wavfile import write as write_wav
import librosa

# ----- paths (ปรับ BASE_DIR ให้ตรงเครื่องคุณ) -----
import platform
if platform.system() == "Windows":
    BASE_DIR = r"C:\Luna-AI-Therapist"
else:
    BASE_DIR = "/mnt/c/Luna-AI-Therapist"
DIALOGUE_ID = 6  # default, overridden by synth_single_utterance parameter

script_dir = os.path.dirname(os.path.abspath(__file__))
core_zonos_path = os.path.join(BASE_DIR, "core_zonos")
sys.path.insert(0, core_zonos_path)

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as ZONOS_DEFAULT_DEVICE

# ถ้าอยากดูค่าดีฟอลต์ไว้ debug ก็ได้
print("Zonos DEFAULT_DEVICE:", ZONOS_DEFAULT_DEVICE)

# 1) กำหนด device ให้ใช้ CUDA ถ้ามี
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Zonos device:", device)

DIALOGUE_DIR = os.path.join(BASE_DIR, "dissonance", "own_script", "dissonance")

# ใช้ ZONOS_INPUT_JSON ถ้ามี; ถ้าไม่มี fallback ไป tmp_directed_zonos_single.json
DEFAULT_INPUT_JSON = os.path.join(BASE_DIR, "dissonance", "own_script", "dialogue_6", f"dialogue_{DIALOGUE_ID}_directed_zonos.json")

OUTPUT_AUDIO_DIR = os.path.join(DIALOGUE_DIR, "voice")

MODEL_ID = "Zonos-v0.1-transformer"
# On Linux (WSL2), use local filesystem for fast model I/O (avoid slow /mnt/c/ 9pfs)
if platform.system() == "Windows":
    ZONOS_MODEL_DIR = os.path.join(core_zonos_path, "models", MODEL_ID)
else:
    import pathlib
    ZONOS_MODEL_DIR = os.path.join(str(pathlib.Path.home()), "luna_models", MODEL_ID)

MALE_VOICE_SAMPLE_PATH = os.path.join(core_zonos_path, "assets", "male_voice_android17.wav")
FEMALE_VOICE_SAMPLE_PATH = os.path.join(core_zonos_path, "assets", "female_voice_android18.wav")

emotion_order = ["Happiness", "Sadness", "Disgust", "Fear", "Surprise", "Anger", "Other", "Neutral"]
MAX_RETRIES = 2            # ลดจาก 3 → 2 (speedup)
MIN_SECONDS_PER_CHAR = 0.025  # default (overridden by synth_single_utterance parameter)


def load_zonos_model():
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    print(f"Loading Zonos model: {MODEL_ID}")
    config_path = os.path.join(ZONOS_MODEL_DIR, "config.json")
    model_weights_path = os.path.join(ZONOS_MODEL_DIR, "model.safetensors")
    zonos_model = Zonos.from_local(config_path, model_weights_path, device=device)
    print("Zonos model loaded.")

    male_embedding_tensor: Optional[torch.Tensor] = None
    female_embedding_tensor: Optional[torch.Tensor] = None

    model_sr = int(zonos_model.autoencoder.sampling_rate)
    print("Model SR:", model_sr)

    try:
        wav_male, sr_male = torchaudio.load(MALE_VOICE_SAMPLE_PATH)
        if sr_male != model_sr:
            wav_male = torchaudio.functional.resample(wav_male, sr_male, model_sr)
            sr_male = model_sr
        male_embedding_tensor = zonos_model.make_speaker_embedding(wav_male.to(device), sr_male)
    except Exception as e:
        print("Male embedding error:", e)

    try:
        wav_female, sr_female = torchaudio.load(FEMALE_VOICE_SAMPLE_PATH)
        if sr_female != model_sr:
            wav_female = torchaudio.functional.resample(wav_female, sr_female, model_sr)
            sr_female = model_sr
        female_embedding_tensor = zonos_model.make_speaker_embedding(wav_female.to(device), sr_female)
    except Exception as e:
        print("Female embedding error:", e)

    return zonos_model, model_sr, male_embedding_tensor, female_embedding_tensor

# -------------------------
# 3) โหลดครั้งเดียวที่ระดับ module (นอกฟังก์ชัน)
# -------------------------
print("Loading Zonos model once at import...")
_ZONOS_MODEL, _MODEL_SR, _MALE_EMB, _FEMALE_EMB = load_zonos_model()
print("Zonos ready.")

def synth_single_utterance(idx: int, input_json: str, dialogue_id: int = 6, prefix: str = "dialogue", min_sec_per_char: float = None):
    """
    idx = dialogue turn (1,2,3,...)
    input_json = path ไป JSON ที่มี directed_utterances (ของ turn นี้)
    min_sec_per_char = optional: dynamic speaking rate from VA (overrides MIN_SECONDS_PER_CHAR)
    """
    print("Using INPUT_JSON:", input_json)
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"INPUT_JSON not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    directed_utterances = data.get("directed_utterances", [])
    print("Total utterances in JSON:", len(directed_utterances))

    if len(directed_utterances) < 1:
        raise IndexError("No directed_utterances in JSON")

    utt = directed_utterances[0]

    zonos_model = _ZONOS_MODEL
    model_sr = _MODEL_SR
    male_emb = _MALE_EMB
    female_emb = _FEMALE_EMB
    selected_speaker_tensor = female_emb  # หรือสลับตามที่ใช้เป็นเสียง client

    text_with_tags = utt.get("utterance_text", "")
    rule = utt.get("new_utterance_rule_definition", {}) or {}
    final_vec = rule.get("primary_zonos_vector_value", {}) or {}

    gen_params = utt.get("generation_parameters", {}) or {}
    speaking_rate = gen_params.get("speaking_rate", rule.get("speaking_rate"))
    pitch_std = gen_params.get("pitch_std", rule.get("pitch_std"))

    if not isinstance(final_vec, dict) or not final_vec:
        print("Skip utterance", idx, "no vector:", final_vec)
        return None

    clean_text = re.sub(r"\[.*?\]", "", text_with_tags).strip()
    if not clean_text:
        print("Skip utterance", idx, "empty text")
        return None

    emotion_list = [final_vec.get(em, 0.0) for em in emotion_order]

    cond_dict = make_cond_dict(
        text=clean_text,
        emotion=emotion_list,
        language="en-us",
        speaking_rate=speaking_rate,
        pitch_std=pitch_std,
        speaker=selected_speaker_tensor,
    )

    # heuristic: dynamic from VA if provided, else module default
    sec_per_char = min_sec_per_char if min_sec_per_char is not None else MIN_SECONDS_PER_CHAR
    expected_min_sec = max(2.0, len(clean_text) * sec_per_char)
    print(f"Expected min duration ~{expected_min_sec:.2f}s (rate={sec_per_char:.4f}) for utterance {idx}")

    MIN_RMS = 0.02  # lower from 0.3 → 0.02: sad/quiet voices are naturally soft

    best_out = None
    best_score = 0.0  # ใช้ score = duration_sec * rms เพื่อเลือกตัวดีที่สุด

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Zonos] Utterance {idx} attempt {attempt}/{MAX_RETRIES}")
            conditioning = zonos_model.prepare_conditioning(cond_dict)
            codes = zonos_model.generate(conditioning, disable_torch_compile=True)
            wav_tensor = zonos_model.autoencoder.decode(codes).cpu()
            if wav_tensor.numel() == 0:
                print(f"Empty audio tensor at utterance {idx}, attempt {attempt}")
                continue

            audio_numpy = (wav_tensor.squeeze().numpy() * 32767).astype(np.int16)

            audio_float = audio_numpy.astype(np.float32) / 32767.0
            if audio_float.size == 0:
                print("Only silence after decode at utterance", idx, "attempt", attempt)
                continue

            # 3-pass audio cleaning pipeline
            # Pass 1: spectral noise reduction
            try:
                import noisereduce as nr
                audio_float = nr.reduce_noise(y=audio_float, sr=model_sr, prop_decrease=0.8)
            except Exception:
                pass  # noisereduce not installed -> skip

            # Pass 2: remove dead air, preserve natural phrase boundaries
            intervals = librosa.effects.split(audio_float, top_db=30)
            if len(intervals) == 0:
                print("Only silence after split at utterance", idx, "attempt", attempt)
                continue
            GAP_MS = 0.200  # 200ms natural pause between phrases
            gap_samples = int(GAP_MS * model_sr)
            segments = []
            for s_start, s_end in intervals:
                segments.append(audio_float[s_start:s_end])
                segments.append(np.zeros(gap_samples, dtype=np.float32))
            y_clean = np.concatenate(segments[:-1]) if len(segments) > 1 else audio_float

            # Pass 3: gentle edge trim (preserves breathy endings, no global normalization)
            y_trimmed, _ = librosa.effects.trim(y_clean, top_db=30)
            if y_trimmed.size == 0:
                print("Only silence after cleaning at utterance", idx, "attempt", attempt)
                continue

            # วัด duration + RMS
            duration_sec = y_trimmed.shape[0] / model_sr
            rms = float(np.sqrt(np.mean(y_trimmed ** 2)))
            score = duration_sec * rms
            print(f"Attempt {attempt}: duration={duration_sec:.2f}s, rms={rms:.3f}")

            # quality check
            audio_out_raw = (y_trimmed * 32767).astype(np.int16)

            # เก็บตัวที่ "ดีที่สุด" ไว้เป็น fallback
            if score > best_score:
                best_score = score
                best_out = (y_trimmed.copy(), model_sr, duration_sec, rms)

            # เงื่อนไขผ่าน
            if duration_sec >= expected_min_sec and rms >= MIN_RMS:
                out_name = f"{prefix}_{dialogue_id}_utterance_{idx}.wav"
                out_path = os.path.join(OUTPUT_AUDIO_DIR, out_name)
                write_wav(out_path, model_sr, audio_out_raw)
                print(
                    f"Saved: {out_path} (dur={duration_sec:.2f}s, rms={rms:.3f}, attempt {attempt})"
                )
                return out_path

        except Exception as e:
            print("Error at utterance", idx, "attempt", attempt, ":", e)

    # fallback: save best-effort audio even if quality check failed
    if best_out is not None:
        y_best, sr, best_dur, best_rms = best_out
        audio_out = (y_best * 32767).astype(np.int16)
        out_name = f"{prefix}_{dialogue_id}_utterance_{idx}.wav"
        out_path = os.path.join(OUTPUT_AUDIO_DIR, out_name)
        write_wav(out_path, sr, audio_out)
        print(
            f"[BEST-EFFORT] Saved utterance {idx} (dur={best_dur:.2f}s, rms={best_rms:.3f})"
        )
        return out_path

    print(f"Failed to synth utterance {idx} after {MAX_RETRIES} retries.")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="dialogue turn index (1-based)")
    args = parser.parse_args()

    synth_single_utterance(args.idx)


# if __name__ == "__main__":
#     main()
