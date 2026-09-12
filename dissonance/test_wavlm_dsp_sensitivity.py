#!/usr/bin/env python3
"""
Diagnostic test #2: Does the WavLM SER model respond to acoustic emotion cues AT ALL?

Takes neutral synthesized audio (ztest_A_neutral_* from test #1, or real dialogue
audio as fallback) and applies graded DSP emotion transformations:

  neutral        - untouched control
  sad_mild       - pitch -2 st, tempo x0.85, energy x0.8
  sad_strong     - pitch -4 st, tempo x0.70, energy x0.6, lowpass 3kHz
  anxious_mild   - pitch +2 st, tempo x1.15, tremolo 8Hz
  anxious_strong - pitch +4 st, tempo x1.35, energy x1.1, tremolo 12Hz + noise

Then measures val_s/aro_s with the EXACT WavLM SER model the pipeline uses.

VERDICT GUIDE:
- sad_strong val_s clearly BELOW neutral  => WavLM IS sensitive => audio-level
  emotion rendering is VIABLE (we fix the leakage channel with DSP when masking)
- nothing moves even under strong DSP     => WavLM SER is the bottleneck =>
  need a different speech-emotion measurement approach
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

def get_acoustics(wav_path):
    y, sr = librosa.load(str(wav_path), sr=None)
    dur = len(y) / sr
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
    f0 = f0[~np.isnan(f0)]
    rms = float(np.sqrt(np.mean(y ** 2)))
    f0_mean = float(np.mean(f0)) if len(f0) else 0.0
    f0_std = float(np.std(f0)) if len(f0) else 0.0
    return f0_mean, f0_std, rms, dur

# =====================================================
# DSP emotion transforms
# =====================================================
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
    if name == "sad_mild":
        out = pitch_shift_st(y, sr, -2)
        out = time_stretch_rate(out, 0.85)
        return out * 0.8
    if name == "sad_strong":
        out = pitch_shift_st(y, sr, -4)
        out = time_stretch_rate(out, 0.70)
        out = lowpass(out, sr, 3000)
        return out * 0.6
    if name == "anxious_mild":
        out = pitch_shift_st(y, sr, +2)
        out = time_stretch_rate(out, 1.15)
        return tremolo(out, sr, freq=8.0, depth=0.30)
    if name == "anxious_strong":
        out = pitch_shift_st(y, sr, +4)
        out = time_stretch_rate(out, 1.35)
        out = tremolo(out, sr, freq=12.0, depth=0.40)
        out = add_noise(out, alpha=0.004)
        return out * 1.1
    raise ValueError(name)

VARIANTS = ["neutral", "sad_mild", "sad_strong", "anxious_mild", "anxious_strong"]

# =====================================================
# Source audio: prefer ztest_A_neutral from test #1, else real dialogue wavs
# =====================================================
sources = sorted(glob.glob(str(VOICE_DIR / "ztest_A_neutral_999_utterance_*.wav")))
if len(sources) < 2:
    print("[WARN] ztest_A_neutral files not found, falling back to real dialogue audio")
    cand = sorted(glob.glob(str(VOICE_DIR / "dissonance_gpt4o-mini_1_utterance_1.wav")))
    cand += sorted(glob.glob(str(VOICE_DIR / "dissonance_gpt4o-mini_1_utterance_2.wav")))
    sources = cand[:2]
if not sources:
    print("[ERROR] no source audio found"); sys.exit(1)
sources = sources[:2]
print(f"Source audio: {[Path(s).name for s in sources]}")

# =====================================================
# Run
# =====================================================
print("=" * 100)
print("WAVLM DSP-SENSITIVITY TEST")
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
        out_path = VOICE_DIR / f"ztest2_{vname}_src{si + 1}.wav"
        write_wav(str(out_path), sr, (np.clip(out, -1, 1) * 32767).astype(np.int16))
        val_s, aro_s = get_speech_VA(out_path)
        f0m, f0s, rms, dur = get_acoustics(out_path)
        results.append((si + 1, vname, val_s, aro_s, f0m, f0s, rms, dur))
        print(f"  {vname:<15} => VA=({val_s:+.2f},{aro_s:+.2f}) F0={f0m:.0f}+/-{f0s:.0f}Hz RMS={rms:.3f} dur={dur:.1f}s")

# =====================================================
# Summary with deltas vs neutral control
# =====================================================
print("\n" + "=" * 100)
print("SUMMARY (delta vs neutral control of the same source)")
print("=" * 100)
print(f"{'src':<4}{'variant':<16}{'val_s':>7}{'aro_s':>7}{'dVal':>7}{'dAro':>7}{'F0':>6}{'RMS':>7}")
for si in [1, 2]:
    base = next((r for r in results if r[0] == si and r[1] == "neutral"), None)
    if base is None:
        continue
    for r in results:
        if r[0] != si:
            continue
        _, vname, val_s, aro_s, f0m, f0s, rms, dur = r
        dval = val_s - base[2]
        daro = aro_s - base[3]
        print(f"{si:<4}{vname:<16}{val_s:>7.2f}{aro_s:>7.2f}{dval:>+7.2f}{daro:>+7.2f}{f0m:>6.0f}{rms:>7.3f}")

print()
print("VERDICT GUIDE:")
print("- sad_strong dVal <= -0.15           => WavLM IS sensitive to acoustic emotion")
print("                                        => audio-level rendering is VIABLE (fix masking with DSP)")
print("- all |dVal| < 0.10 even for strong  => WavLM SER is the bottleneck")
print("                                        => need different speech-emotion measurement")
print()
print("NOTE: transformed wavs saved as ztest2_* in dissonance/own_script/dissonance/voice/")
print("      safe to delete: rm .../voice/ztest2_*")
