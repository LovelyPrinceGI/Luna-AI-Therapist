#!/usr/bin/env python3
"""
Emotion DSP leakage channel for Luna multibackbone pipelines.

Zonos-v0.1 emotion-vector conditioning does NOT render measurable affect
(verified by test_zonos_emotion_conditioning.py: corr(zonos_neg, val_s) ~ 0.02).
However, the WavLM SER model IS sensitive to acoustic emotion cues
(verified by test_wavlm_dsp_sensitivity.py: sad DSP => dVal -0.42 / -0.24).

DSP tests #3/#4 established the acoustic rule for this SER model:
  - NEGATIVE valence is read from LOW-pitch / slower / quieter speech.
  - Pitch-UP "anxious" variants (with or without tremolo) do NOT produce
    negative valence; tremolo adds no benefit.

This module therefore renders the client's INTENDED (possibly hidden) NEGATIVE
emotion into the synthesized audio via a unified graded "distress" transform
(Sadness / Fear / Anger / Disgust all leak as heavy, pressed, low-energy voice):

  strong (mag >= 0.75): pitch -4 st, tempo x0.70, energy x0.6, lowpass 3000
                        (measured mean dVal ~ -0.33)
  mild (0.50 <= mag < 0.75): pitch -2 st, tempo x0.85, energy x0.8
                        (measured mean dVal ~ -0.09)

Which turns get DSP, which emotion, and which strength are all driven by the
client's own zonos_director output (nothing is injected arbitrarily).

Applied to the client audio BEFORE val_s / vocal-descriptor measurement, so the
voice carries the client's true inner state:
  - honest turns  -> words and voice agree (no false dissonance)
  - masked turns  -> words neutral, voice negative => real dissonance signal
"""
import numpy as np
import librosa
import soundfile as sf
from scipy.io.wavfile import write as write_wav
from scipy.signal import butter, sosfilt

EMOTION_DSP_THRESHOLD = 0.5
EMOTION_DSP_STRONG = 0.75
NEG_EMOTIONS = ["Sadness", "Fear", "Anger", "Disgust"]


def _lowpass(y, sr, cutoff):
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, y)


def _distress_dsp(y, sr, strong):
    if strong:
        out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-4.0)
        out = librosa.effects.time_stretch(y=out, rate=0.70)
        out = _lowpass(out, sr, 3000)
        return out * 0.6
    out = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2.0)
    out = librosa.effects.time_stretch(y=out, rate=0.85)
    return out * 0.8


def pick_emotion_dsp(zonos_params):
    """
    Decide DSP kind from the client's zonos_director vector.
    Returns (kind_string, dominant_emotion, magnitude) or (None, None, 0.0).
    kind_string in {distress_mild, distress_strong}.
    """
    if not zonos_params:
        return None, None, 0.0
    vec = zonos_params.get("primary_zonos_vector_value") or {}
    if not vec:
        return None, None, 0.0
    clamped = {k: max(0.0, min(1.0, float(v))) for k, v in vec.items()}
    dom_emotion, mag = max(((k, clamped.get(k, 0.0)) for k in NEG_EMOTIONS), key=lambda x: x[1])
    if mag < EMOTION_DSP_THRESHOLD:
        return None, dom_emotion, mag
    strength = "strong" if mag >= EMOTION_DSP_STRONG else "mild"
    return f"distress_{strength}", dom_emotion, mag


def apply_emotion_dsp(wav_path, zonos_params):
    """
    Render the client's intended emotion into the audio file (in place).
    Returns the applied kind string, or None if no DSP was applied.
    """
    kind, dom_emotion, mag = pick_emotion_dsp(zonos_params)
    if kind is None:
        return None

    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if len(y) < sr * 0.5:
        return None

    out = _distress_dsp(y, sr, strong=(kind == "distress_strong"))

    out = np.clip(out, -1.0, 1.0)
    write_wav(str(wav_path), sr, (out * 32767).astype(np.int16))
    print(f"  [EMOTION-DSP] applied {kind} ({dom_emotion}={mag:.2f})")
    return kind
