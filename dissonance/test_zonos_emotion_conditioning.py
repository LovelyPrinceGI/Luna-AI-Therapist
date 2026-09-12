#!/usr/bin/env python3
"""
Standalone test: Does Zonos actually render masked emotion into audio?

Synthesizes the SAME neutral-valence text under different zonos conditioning
variants and measures:
  1. Acoustics (F0 mean/std, RMS, words/sec)
  2. WavLM SER valence/arousal (the EXACT model the pipeline uses)

Variants (per text):
  A_neutral         - honest baseline
  B_masked_current  - what the client outputs TODAY (negative values, sum~0.3)
  C_masked_clamped  - same intent, emotion clamped to 0..1
  D_clamped_pitch   - clamped + expressive pitch_std (110)
  E_masked_anxious  - clamped fear + fast rate + high pitch_std
  F_happy_clamped   - positive control (should read positive)

Output files are named ztest_* (never overwrite real dialogue audio).

HOW TO READ THE RESULTS:
- If B ~ A            -> current client output renders NO emotion (broken signal)
- If C/D/E < A clearly -> clamping (+pitch_std) FIXES the leakage channel
- If F > A            -> positive direction also works
"""
import json
import sys
from pathlib import Path
import platform

if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

SYNTH_DIR = BASE_DIR / "dissonance" / "own_script" / "dialogue_6"
sys.path.insert(0, str(SYNTH_DIR))
from run_synthesis_dialogue_6_module import synth_single_utterance  # loads Zonos once

import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import AutoModelForAudioClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Test device: {device}")

TMP_JSON = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "tmp_zonos_test.json"

# =====================================================
# WavLM SER — identical to run_dissonance_multibackbone_v3.py
# =====================================================
WAVLM_MODEL_NAME = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"
print(f"Loading WavLM: {WAVLM_MODEL_NAME} ...")
_wavlm = AutoModelForAudioClassification.from_pretrained(WAVLM_MODEL_NAME, trust_remote_code=True).to(device)
_wavlm.eval()
_target_sr = _wavlm.config.sampling_rate
_mean = _wavlm.config.mean
_std = _wavlm.config.std

def get_speech_VA(wav_path):
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_target_sr)
    audio = (audio - _mean) / (_std + 1e-6)
    wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, wavs.shape[1], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = _wavlm(wavs, mask)
    logits = pred.cpu().numpy()[0].astype(float)
    aro, dom, val = float(logits[0]), float(logits[1]), float(logits[2])
    return (2.0 * val - 1.0), (2.0 * aro - 1.0)

def get_acoustics(wav_path, text):
    y, sr = librosa.load(str(wav_path), sr=None)
    dur = len(y) / sr
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    f0 = f0[~np.isnan(f0)]
    rms = float(np.sqrt(np.mean(y ** 2)))
    f0_mean = float(np.mean(f0)) if len(f0) else 0.0
    f0_std = float(np.std(f0)) if len(f0) else 0.0
    words_per_sec = len(text.split()) / dur if dur > 0 else 0.0
    return f0_mean, f0_std, rms, words_per_sec, dur

# =====================================================
# Test matrix
# =====================================================
TEST_TEXTS = [
    "I'm managing fine, really. It's just a new job, and everyone deals with deadlines.",
    "It's not that big of a deal. I'm okay, I just need to keep moving forward.",
]

def vec(H, S, D, F, Su, A, O, N):
    return {"Happiness": H, "Sadness": S, "Disgust": D, "Fear": F,
            "Surprise": Su, "Anger": A, "Other": O, "Neutral": N}

VARIANTS = [
    # name, emotion vector, speaking_rate, pitch_std
    ("A_neutral",        vec(0.1, 0.05, 0.0, 0.05, 0.0, 0.0, 0.0, 0.8), 16.0, 40.0),
    ("B_masked_current", vec(-0.7, 0.9, -0.2, 0.7, -0.3, 0.5, 0.0, -0.6), 12.0, 45.0),
    ("C_masked_clamped", vec(0.0, 0.9, 0.0, 0.7, 0.0, 0.5, 0.0, 0.0), 12.0, 45.0),
    ("D_clamped_pitch",  vec(0.0, 0.9, 0.0, 0.7, 0.0, 0.5, 0.0, 0.0), 12.0, 110.0),
    ("E_masked_anxious", vec(0.0, 0.2, 0.0, 0.9, 0.3, 0.2, 0.0, 0.0), 24.0, 130.0),
    ("F_happy_clamped",  vec(0.9, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0), 18.0, 90.0),
]

def synth_variant(name, text, emotion_vec, rate, pstd, idx):
    directed = {
        "utterance_text": text,
        "is_new_utterance_rule": True,
        "utterance_level_direction": "test",
        "new_utterance_rule_definition": {
            "primary_zonos_vector_value": emotion_vec,
            "speaking_rate": rate,
            "pitch_std": pstd,
        },
        "frame_level_directions": [],
    }
    TMP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with TMP_JSON.open("w", encoding="utf-8") as f:
        json.dump({"directed_utterances": [directed]}, f, ensure_ascii=False, indent=2)
    return synth_single_utterance(idx, str(TMP_JSON), dialogue_id=999,
                                  prefix=f"ztest_{name}", min_sec_per_char=0.030)

# =====================================================
# Run
# =====================================================
print("=" * 100)
print("ZONOS EMOTION CONDITIONING TEST")
print("=" * 100)

results = []
idx = 0
for ti, text in enumerate(TEST_TEXTS):
    for vname, vvec, vrate, vpstd in VARIANTS:
        idx += 1
        print(f"\n>>> text{ti + 1} | {vname} | rate={vrate} pitch_std={vpstd} | vec={vvec}")
        wav = synth_variant(vname, text, vvec, vrate, vpstd, idx)
        if wav is None:
            print("  SYNTH FAILED")
            continue
        val_s, aro_s = get_speech_VA(wav)
        f0m, f0s, rms, wps, dur = get_acoustics(wav, text)
        results.append((ti + 1, vname, val_s, aro_s, f0m, f0s, rms, wps, dur))
        print(f"  => VA=({val_s:+.2f},{aro_s:+.2f}) F0={f0m:.0f}+/-{f0s:.0f}Hz "
              f"RMS={rms:.3f} words/s={wps:.1f} dur={dur:.1f}s")

# =====================================================
# Summary
# =====================================================
print("\n" + "=" * 100)
print("SUMMARY  (val_s is the key number: negative = sad/anxious voice, positive = happy)")
print("=" * 100)
print(f"{'text':<5}{'variant':<18}{'val_s':>7}{'aro_s':>7}{'F0':>6}{'F0sd':>6}{'RMS':>7}{'w/s':>6}")
for ti, vname, val_s, aro_s, f0m, f0s, rms, wps, dur in results:
    print(f"{ti:<5}{vname:<18}{val_s:>7.2f}{aro_s:>7.2f}{f0m:>6.0f}{f0s:>6.0f}{rms:>7.3f}{wps:>6.1f}")

print("\n--- average per variant (across texts) ---")
from collections import defaultdict
by_variant = defaultdict(list)
for ti, vname, val_s, aro_s, *_ in results:
    by_variant[vname].append(val_s)
for vname, *_ in VARIANTS:
    vals = by_variant.get(vname, [])
    if vals:
        print(f"{vname:<18} mean val_s = {sum(vals) / len(vals):+.2f}")

print()
print("VERDICT GUIDE:")
print("- B_masked_current ~= A_neutral      => current output renders NO emotion (signal broken)")
print("- C/D/E clearly more negative than A  => clamp (+pitch_std) FIXES the leakage channel")
print("- F_happy_clamped > A_neutral         => positive direction also controllable")
print()
print("NOTE: test wavs saved as ztest_* in dissonance/own_script/dissonance/voice/")
print("      safe to delete after inspection: rm .../voice/ztest_*")
