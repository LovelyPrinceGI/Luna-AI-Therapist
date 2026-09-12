#!/usr/bin/env python3
"""
DSP test #3: find an ANXIOUS variant that pushes WavLM valence NEGATIVE.

Finding from test #2: anxious_strong (pitch+4, tempo x1.35, energy x1.1)
barely moved valence (+0.16 / -0.12). High energy + very fast reads as
"excited/activated" rather than negative. Hypothesis: to make anxiety sound
NEGATIVE we should lower the energy (strained/quiet/tense) instead of boosting
it, while keeping pitch up + fast + shaky.

Variants tested (all start from the same neutral source):
  neutral        - control
  anxious_strong - the CURRENT anxious_strong (reference)
  strained       - pitch+2, tempo x1.2, tremolo, energy x0.85, light noise
  tense_quiet    - pitch+3, tempo x1.25, tremolo, energy x0.7, noise
  shaky_low      - pitch+2, tempo x1.15, strong tremolo, energy x0.75, lowpass
  dread          - pitch+1, tempo x1.1, tremolo+noise, energy x0.65, lowpass

Lower mean dVal (vs neutral) = more negative = better anxious leakage.
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

def add_noise(y, alpha=0.004, seed=0):
    rng = np.random.default_rng(seed)
    return y + alpha * rng.standard_normal(len(y))

def pitch_shift_st(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=float(n_steps))

def time_stretch_rate(y, rate):
    return librosa.effects.time_stretch(y=y, rate=float(rate))

def make_variant(name, y, sr):
    if name == "neutral":
        return y
    if name == "anxious_strong":      # current implementation (reference)
        out = pitch_shift_st(y, sr, +4)
        out = time_stretch_rate(out, 1.35)
        out = tremolo(out, sr, freq=12.0, depth=0.40)
        out = add_noise(out, alpha=0.004)
        return out * 1.1
    if name == "strained":
        out = pitch_shift_st(y, sr, +2)
        out = time_stretch_rate(out, 1.20)
        out = tremolo(out, sr, freq=10.0, depth=0.40)
        out = add_noise(out, alpha=0.003)
        return out * 0.85
    if name == "tense_quiet":
        out = pitch_shift_st(y, sr, +3)
        out = time_stretch_rate(out, 1.25)
        out = tremolo(out, sr, freq=12.0, depth=0.45)
        out = add_noise(out, alpha=0.004)
        return out * 0.70
    if name == "shaky_low":
        out = pitch_shift_st(y, sr, +2)
        out = time_stretch_rate(out, 1.15)
        out = tremolo(out, sr, freq=8.0, depth=0.50)
        out = lowpass(out, sr, 5000)
        return out * 0.75
    if name == "dread":
        out = pitch_shift_st(y, sr, +1)
        out = time_stretch_rate(out, 1.10)
        out = tremolo(out, sr, freq=9.0, depth=0.50)
        out = add_noise(out, alpha=0.005)
        out = lowpass(out, sr, 4000)
        return out * 0.65
    if name == "tight":                # constricted throat: no pitch shift
        out = time_stretch_rate(y, 1.30)
        out = tremolo(out, sr, freq=12.0, depth=0.40)
        out = lowpass(out, sr, 2500)
        return out * 0.75
    raise ValueError(name)

VARIANTS = ["neutral", "anxious_strong", "strained", "tense_quiet", "shaky_low", "dread", "tight"]

sources = sorted(glob.glob(str(VOICE_DIR / "ztest_A_neutral_999_utterance_*.wav")))[:2]
if not sources:
    print("[ERROR] ztest_A_neutral_999_utterance_*.wav not found; run test #1 first")
    sys.exit(1)
print(f"Source audio: {[Path(s).name for s in sources]}")

print("=" * 100)
print("ANXIOUS-DSP VARIANT TEST")
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
        out_path = VOICE_DIR / f"ztest3_{vname}_src{si + 1}.wav"
        write_wav(str(out_path), sr, (np.clip(out, -1, 1) * 32767).astype(np.int16))
        val_s, aro_s = get_speech_VA(out_path)
        results.append((si + 1, vname, val_s, aro_s))
        print(f"  {vname:<15} => VA=({val_s:+.2f},{aro_s:+.2f})")

print("\n" + "=" * 100)
print("SUMMARY (dVal vs neutral control of same source; MORE NEGATIVE = better anxious)")
print("=" * 100)
print(f"{'src':<4}{'variant':<16}{'val_s':>7}{'aro_s':>7}{'dVal':>7}{'dAro':>7}")
for si in [1, 2]:
    base = next((r for r in results if r[0] == si and r[1] == "neutral"), None)
    if base is None:
        continue
    for r in results:
        if r[0] != si:
            continue
        _, vname, val_s, aro_s = r
        print(f"{si:<4}{vname:<16}{val_s:>7.2f}{aro_s:>7.2f}{val_s - base[2]:>+7.2f}{aro_s - base[3]:>+7.2f}")

print("\n--- mean dVal per variant (across sources) ---")
from collections import defaultdict
dvals = defaultdict(list)
for si in [1, 2]:
    base = next((r for r in results if r[0] == si and r[1] == "neutral"), None)
    if base is None:
        continue
    for r in results:
        if r[0] == si and r[1] != "neutral":
            dvals[r[1]].append(r[2] - base[2])
ranked = sorted(dvals.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
for vname, vals in ranked:
    print(f"{vname:<16} mean dVal = {sum(vals) / len(vals):+.3f}")
print(f"\nBEST anxious variant (most negative): {ranked[0][0]}")
print("NOTE: wavs saved as ztest3_* (safe to delete: rm .../voice/ztest3_*)")
