import torch
import torchaudio
import os
import json
import re
from tqdm import tqdm
import sys
import numpy as np
from scipy.io.wavfile import write as write_wav

# --- Add 'core_zonos' to the path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
core_zonos_path = os.path.join(script_dir, 'core_zonos')
sys.path.insert(0, core_zonos_path)

# --- Imports based on your REAL sample.py ---
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# --- 1. Configuration ---
INPUT_JSONL_FILE = "datasets/final_hierarchical_output_v11_d20.jsonl" 
MODEL_ID = "Zonos-v0.1-transformer" 
ZONOS_MODEL_DIR = os.path.join("core_zonos", "models", MODEL_ID)
OUTPUT_AUDIO_DIR = "final_audio_output_final" 

# --- 2. Main Synthesis Function ---
def synthesize_audio():
    # --- The CORRECT Model Loading Logic ---
    print(f"Loading Zonos model: {MODEL_ID}")
    config_path = os.path.join(ZONOS_MODEL_DIR, 'config.json')
    model_weights_path = os.path.join(ZONOS_MODEL_DIR, 'model.safetensors') 
    
    if not os.path.exists(config_path) or not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Ensure 'config.json' and 'model.safetensors' exist in {ZONOS_MODEL_DIR}")

    zonos_model = Zonos.from_local(config_path, model_weights_path, device=device)
    print("Zonos model loaded successfully!")
    
    # --- The rest of the script ---
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)

    with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"Found {len(lines)} dialogues to process.")

    # นี่คือลำดับที่ Zonos คาดหวัง (ต้องตรงกับตอนเทรน)
    emotion_order = ["Happiness", "Sadness", "Disgust", "Fear", "Surprise", "Anger", "Other", "Neutral"]

    for dialogue_index, line in enumerate(tqdm(lines, desc="Processing Dialogues")):
        data = json.loads(line.strip())
        directed_utterances = data.get("llm_output", {}).get("directed_utterances", [])

        for utterance_index, utterance_data in enumerate(tqdm(directed_utterances, desc=f"  Dialogue {dialogue_index+1}", leave=False)):
            utterance_text_with_tags = utterance_data.get("utterance_text", "")
            final_zonos_vector_dict = utterance_data.get("final_zonos_vector", {})
            
            # --- NEW: ดึง Generation Parameters ออกมา ---
            gen_params = utterance_data.get("generation_parameters", {})
            speaking_rate = gen_params.get("speaking_rate") # จะเป็น None ถ้าไม่มี
            pitch_std = gen_params.get("pitch_std")         # จะเป็น None ถ้าไม่มี
            # --- END NEW ---

            # ทำความสะอาด text (เอาร [sighs] ออก)
            clean_text = re.sub(r'\[.*?\]', '', utterance_text_with_tags).strip()
            
            if not clean_text or not final_zonos_vector_dict: 
                continue # ข้ามถ้าไม่มีบทพูดหรือ vector

            # สร้าง emotion list ตามลำดับที่ถูกต้อง
            emotion_list = [final_zonos_vector_dict.get(emotion, 0.0) for emotion in emotion_order]
            
            # --- UPDATED: ส่ง parameters ทั้งหมดให้ Zonos ---
            # Zonos (make_cond_dict) ฉลาดพอที่จะจัดการกับค่าที่เป็น None ค่ะ
            cond_dict = make_cond_dict(
                text=clean_text, 
                emotion=emotion_list, 
                language='en-us',
                speaking_rate=speaking_rate, # เพิ่ม speaking_rate
                pitch_std=pitch_std          # เพิ่ม pitch_std
            )
            # --- END UPDATED ---

            conditioning = zonos_model.prepare_conditioning(cond_dict)
            codes = zonos_model.generate(conditioning, disable_torch_compile=True)
            wav_tensor = zonos_model.autoencoder.decode(codes).cpu()

            sample_rate = int(zonos_model.autoencoder.sampling_rate)
            audio_numpy = (wav_tensor[0, 0].numpy() * 32767).astype(np.int16)
            
            output_filename = f"dialogue_{dialogue_index+1}_utterance_{utterance_index+1}.wav"
            output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
            write_wav(output_path, sample_rate, audio_numpy)

if __name__ == "__main__":
    synthesize_audio()
    print(f"\n--- Synthesis Complete! ---")
    print(f"All audio files have been saved in the '{OUTPUT_AUDIO_DIR}' directory.")