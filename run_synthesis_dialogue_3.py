import torch
import torchaudio
import os
import json
import re
import sys
import numpy as np
from scipy.io.wavfile import write as write_wav
from typing import Optional
import librosa
import math

# ----- paths -----
script_dir = os.path.dirname(os.path.abspath(__file__))
core_zonos_path = os.path.join(script_dir, "core_zonos")
sys.path.insert(0, core_zonos_path)

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

BASE_DIR = "/home/patsakornt/work/test"
DIALOGUE_ID = 3
DIALOGUE_DIR = os.path.join(BASE_DIR, "dissonance", "own_script", f"dialogue_{DIALOGUE_ID}")
INPUT_JSON = os.path.join(DIALOGUE_DIR, f"dialogue_{DIALOGUE_ID}_directed_zonos.json")
OUTPUT_AUDIO_DIR = os.path.join(DIALOGUE_DIR, "voice")

MODEL_ID = "Zonos-v0.1-transformer"
ZONOS_MODEL_DIR = os.path.join("core_zonos", "models", MODEL_ID)

MALE_VOICE_SAMPLE_PATH = os.path.join("core_zonos", "assets", "male_voice_android17.wav")
FEMALE_VOICE_SAMPLE_PATH = os.path.join("core_zonos", "assets", "female_voice_android18.wav")

def main():
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    print(f"Loading Zonos model: {MODEL_ID}")
    config_path = os.path.join(ZONOS_MODEL_DIR, "config.json")
    model_weights_path = os.path.join(ZONOS_MODEL_DIR, "model.safetensors")
    zonos_model = Zonos.from_local(config_path, model_weights_path, device=device)
    print("Zonos model loaded.")

    # speaker embeddings (ใช้แบบเดียวกับสคริปต์หลัก)
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

    # ----- load directed_utterances for this dialogue -----
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    directed_utterances = data.get("directed_utterances", [])
    print("Utterances:", len(directed_utterances))

    # สำหรับ client ใน dataset นี้ อยากใช้เสียงเพศไหน เลือกตรงนี้
    selected_speaker_tensor = female_embedding_tensor  # หรือ male_embedding_tensor

    emotion_order = ["Happiness", "Sadness", "Disgust", "Fear", "Surprise", "Anger", "Other", "Neutral"]


    
    MAX_RETRIES = 2          # ลอง gen ซ้ำได้สูงสุด 2 รอบ
    MIN_SECONDS_PER_CHAR = 0.025  # heuristic: ตัวอักษรละ ~0.025 วิ
    
    for idx, utt in enumerate(directed_utterances, start=1):
        text_with_tags = utt.get("utterance_text", "")
        rule = utt.get("new_utterance_rule_definition", {}) or {}
        final_vec = rule.get("primary_zonos_vector_value", {}) or {}
    
        gen_params = utt.get("generation_parameters", {}) or {}
        speaking_rate = gen_params.get("speaking_rate", rule.get("speaking_rate"))
        pitch_std = gen_params.get("pitch_std", rule.get("pitch_std"))
    
        if not isinstance(final_vec, dict) or not final_vec:
            print("Skip utterance", idx, "no vector:", final_vec)
            continue
    
        clean_text = re.sub(r"\[.*?\]", "", text_with_tags).strip()
        if not clean_text:
            print("Skip utterance", idx, "empty text")
            continue
    
        emotion_list = [final_vec.get(em, 0.0) for em in emotion_order]
    
        cond_dict = make_cond_dict(
            text=clean_text,
            emotion=emotion_list,
            language="en-us",
            speaking_rate=speaking_rate,
            pitch_std=pitch_std,
            speaker=selected_speaker_tensor,
        )
    
        # ประเมินความยาวขั้นต่ำตามจำนวนตัวอักษร
        expected_min_sec = max(2.0, len(clean_text) * MIN_SECONDS_PER_CHAR)
    
        for attempt in range(1, MAX_RETRIES + 2):  # n retries = MAX_RETRIES
            try:
                conditioning = zonos_model.prepare_conditioning(cond_dict)
                codes = zonos_model.generate(conditioning, disable_torch_compile=True)
                wav_tensor = zonos_model.autoencoder.decode(codes).cpu()
                if wav_tensor.numel() == 0:
                    print(f"Empty audio at utterance {idx}, attempt {attempt}")
                    continue
    
                audio_numpy = (wav_tensor.squeeze().numpy() * 32767).astype(np.int16)
    
                # ตัด silence รอบขอบ
                audio_float = audio_numpy.astype(np.float32) / 32767.0
                y_trimmed, _ = librosa.effects.trim(audio_float, top_db=40)
                if y_trimmed.size == 0:
                    print("Only silence at utterance", idx, "attempt", attempt)
                    continue
    
                audio_out = (y_trimmed * 32767).astype(np.int16)
                duration_sec = audio_out.shape[0] / model_sr
    
                if duration_sec + 1e-3 < expected_min_sec and attempt <= MAX_RETRIES:
                    print(
                        f"Utterance {idx} too short after trim "
                        f"({duration_sec:.2f}s < {expected_min_sec:.2f}s), "
                        f"retry {attempt}/{MAX_RETRIES}"
                    )
                    continue  # ลอง gen ใหม่
                else:
                    out_name = f"dialogue_{DIALOGUE_ID}_utterance_{idx}.wav"
                    out_path = os.path.join(OUTPUT_AUDIO_DIR, out_name)
                    write_wav(out_path, model_sr, audio_out)
                    print(
                        f"Saved: {out_path} (trimmed duration {duration_sec:.2f}s, attempt {attempt})"
                    )
                    break  # ออกจาก loop retry สำหรับ utterance นี้
    
            except Exception as e:
                print("Error at utterance", idx, "attempt", attempt, ":", e)


if __name__ == "__main__":
    main()
