#!/usr/bin/env python3
"""
DSP test #4: pitch-DOWN anxious variants.

Findings from tests #2/#3:
  - WavLM reads negative valence from LOW-pitch / quiet / slow acoustics
    (sad_strong: dVal -0.42 / -0.24)
  - pitch-UP "anxious" variants never produce negative valence

So masked anxiety must leak through pitch-DOWN rendering (heavy/pressed voice)
while tremolo keeps the unsettled character. Candidates:

  neutral      - control
  anx_low_v1   - pitch -2, tempo x0.90, tremolo 10Hz, energy x0.70
  anx_low_v2   - pitch -3, tempo x0.85, tremolo 12Hz, energy x0.60, lowpass 3500
  anx_low_v3   - pitch -1, no stretch, tremolo 10Hz, energy x0.75
  sad_strong   - reference (known strong negative)
  strained     - reference (winner of test #3)

Most negative mean dVal wins -> becomes the anxious DSP in emotion_dsp.py.
"""
import glob
import sys
from pathlib import Path
import platform

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, sosfilt
from transformers import AutoModelForAudioClassification

if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

VOICE_DIR = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "voice"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Test device: {device}")

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

def lowpass(y, sr, cutoff):
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, y)

def tremolo(y, sr, freq=10.0, depth=0.35):
    t = np.arange(len(y)) / sr
    return y * (1.0 - depth * 0.5 * (1.0 + np.sin(2 * np.pi * freq * t)))

def ps(y, sr, steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(steps))

def ts(y, rate):
    return librosa.effects.time_stretch(y=y, rate=float(rate))

def make_variant(name, y, sr):
    if name == "neutral":
        return y
    if name == "anx_low_v1":
        out = ps(y, sr, -2)
        out = ts(out, 0.90)
        out = tremolo(out, sr, freq=10.0, depth=0.40)
        return out * 0.70
    if name == "anx_low_v2":
        out = ps(y, sr, -3)
        out = ts(out, 0.85)
        out = tremolo(out, sr, freq=12.0, depth=0.45)
        out = lowpass(out, sr, 3500)
        return out * 0.60
    if name == "anx_low_v3":
        out = ps(y, sr, -1)
        out = tremolo(out, sr, freq=10.0, depth=0.35)
        return out * 0.75
    if name == "sad_strong":
        out = ps(y, sr, -4)
        out = ts(out, 0.70)
        out = lowpass(out, sr, 3000)
        return out * 0.6
    if name == "strained":
        out = ps(y, sr, +2)
        out = ts(out, 1.20)
        out = tremolo(out, sr, freq=10.0, depth=0.40)
        return out * 0.85
    raise ValueError(name)

VARIANTS = ["neutral", "anx_low_v1", "anx_low_v2", "anx_low_v3", "sad_strong", "strained"]

sources = sorted(glob.glob(str(VOICE_DIR / "ztest_A_neutral_999_utterance_*.wav")))[:2]
if len(sources) < 2:
    print("[ERROR] need ztest_A_neutral_999_utterance_*.wav from test #1"); sys.exit(1)
print(f"Source audio: {[Path(s).name for s in sources]}")

print("=" * 100)
print("PITCH-DOWN ANXIOUS DSP TEST")
print("=" * 100)

results = []
for si, src in enumerate(sources):
    y, sr = sf.read(src)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    print(f"\n>>> source {si + 1}: {Path(src).name} (sr={sr}, dur={len(y)/sr:.1f}s)")
    for vname in VARIANTS:
        out = make_variant(vname, y, sr)
        out_path = VOICE_DIR / f"ztest4_{vname}_src{si + 1}.wav"
        write_wav(str(out_path), sr, (np.clip(out, -1, 1) * 32767).astype(np.int16))
        val_s, aro_s = get_speech_VA(out_path)
        results.append((si + 1, vname, val_s, aro_s))
        print(f"  {vname:<13} => VA=({val_s:+.2f},{aro_s:+.2f})")

print("\n" + "=" * 100)
print("SUMMARY (dVal/dAro vs neutral control of same source; more negative dVal = better)")
print("=" * 100)
print(f"{'src':<4}{'variant':<15}{'val_s':>7}{'aro_s':>7}{'dVal':>7}{'dAro':>7}")
for si in [1, 2]:
    base = next(r for r in results if r[0] == si and r[1] == "neutral")
    for r in results:
        if r[0] != si:
            continue
        _, vname, val_s, aro_s = r
        print(f"{si:<4}{vname:<15}{val_s:>7.2f}{aro_s:>7.2f}{val_s - base[2]:>+7.2f}{aro_s - base[3]:>+7.2f}")

print("\n--- mean dVal per variant (across sources) ---")
from collections import defaultdict
dval_by_variant = defaultdict(list)
for si in [1, 2]:
    base = next(r for r in results if r[0] == si and r[1] == "neutral")
    for r in results:
        if r[0] == si and r[1] != "neutral":
            dval_by_variant[r[1]].append(r[2] - base[2])
ranked = sorted(dval_by_variant.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
for vname, vals in ranked:
    print(f"{vname:<15} mean dVal = {sum(vals) / len(vals):+.3f}")
print(f"\nBEST (most negative): {ranked[0][0]}")
print("NOTE: wavs saved as ztest4_* (safe to delete: rm .../voice/ztest4_*)")
