import torch
import torchaudio
import os
import json
import re
from tqdm import tqdm
import sys
import numpy as np
from scipy.io.wavfile import write as write_wav
from typing import Union, Optional 

# --- Add 'core_zonos' to the path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
core_zonos_path = os.path.join(script_dir, 'core_zonos')
sys.path.insert(0, core_zonos_path)

# --- Imports based on Zonos ---
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# --- 1. Configuration ---
INPUT_JSONL_FILE = "datasets/my_own.jsonl" 
MODEL_ID = "Zonos-v0.1-transformer" 
ZONOS_MODEL_DIR = os.path.join("core_zonos", "models", MODEL_ID)
OUTPUT_AUDIO_DIR = "audio_myown" 

# --- UPDATED: Only Keep Female/Android 18 Path ---
FEMALE_VOICE_SAMPLE_PATH = os.path.join("core_zonos", "assets", "female_voice_android18.wav")

# --- 2. Main Synthesis Function ---
def synthesize_audio():
    # --- Model Loading Logic ---
    print(f"Loading Zonos model: {MODEL_ID}")
    config_path = os.path.join(ZONOS_MODEL_DIR, 'config.json')
    model_weights_path = os.path.join(ZONOS_MODEL_DIR, 'model.safetensors')
    
    if not os.path.exists(config_path) or not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Ensure 'config.json' and 'model.safetensors' exist in {ZONOS_MODEL_DIR}")

    zonos_model = Zonos.from_local(config_path, model_weights_path, device=device)
    print("Zonos model loaded successfully!")

    # --- Load sample audios and create embeddings ---
    print("Creating speaker embeddings (Android 18 ONLY)...")
    speaker_embedding_tensor: Optional[torch.Tensor] = None

    # Determine the model's expected sampling rate
    model_sampling_rate = int(zonos_model.autoencoder.sampling_rate)
    print(f"Model expects audio at {model_sampling_rate} Hz.")

    # --- LOAD ONLY FEMALE VOICE ---
    try:
        wav_female, sr_female = torchaudio.load(FEMALE_VOICE_SAMPLE_PATH)
        # Resample if necessary
        if sr_female != model_sampling_rate:
            print(f"Resampling female audio from {sr_female} Hz to {model_sampling_rate} Hz...")
            wav_female = torchaudio.functional.resample(wav_female, sr_female, model_sampling_rate)
            sr_female = model_sampling_rate 
        
        # Create Embedding
        speaker_embedding_tensor = zonos_model.make_speaker_embedding(wav_female.to(device), sr_female)
        print(f"Successfully created Android 18 speaker embedding from '{FEMALE_VOICE_SAMPLE_PATH}'")
        
    except FileNotFoundError:
        print(f"Warning: Voice sample file not found at '{FEMALE_VOICE_SAMPLE_PATH}'. Will use default random speaker.")
    except Exception as e:
        print(f"Error creating embedding: {e}")
    
    # --- Output Directory & Read Input ---
    if not os.path.exists(OUTPUT_AUDIO_DIR):
        os.makedirs(OUTPUT_AUDIO_DIR)
    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Found {len(lines)} dialogues to process.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_JSONL_FILE}' not found.")
        return

    # --- Emotion Order ---
    emotion_order = ["Happiness", "Sadness", "Disgust", "Fear", "Surprise", "Anger", "Other", "Neutral"]

    # --- Process Each Dialogue ---
    for dialogue_index, line in enumerate(tqdm(lines, desc="Processing Dialogues")):
        try:
            data = json.loads(line.strip())
            
            # --- REMOVED: Gender check logic. Always use speaker_embedding_tensor ---
            
            directed_utterances = data.get("llm_output", {}).get("directed_utterances", [])
            # Handle potential error in llm_output
            if isinstance(directed_utterances, str) and "error" in directed_utterances.lower():
                 print(f"Warning: Skipping Dialogue {dialogue_index+1} due to error in 'llm_output'")
                 continue
            if not isinstance(directed_utterances, list):
                 continue

            # --- Process Each Utterance ---
            for utterance_index, utterance_data in enumerate(tqdm(directed_utterances, desc=f"  Dialogue {dialogue_index+1}", leave=False)):
                 if not isinstance(utterance_data, dict):
                     continue

                 utterance_text_with_tags = utterance_data.get("utterance_text", "")
                 final_zonos_vector_dict = utterance_data.get("final_zonos_vector", {})
                 gen_params = utterance_data.get("generation_parameters", {})
                 speaking_rate = gen_params.get("speaking_rate")
                 pitch_std = gen_params.get("pitch_std")
                 
                 if not isinstance(final_zonos_vector_dict, dict):
                      continue
                 
                 clean_text = re.sub(r'\[.*?\]', '', utterance_text_with_tags).strip()
                 
                 if not clean_text or not final_zonos_vector_dict: 
                     continue 
                     
                 emotion_list = [final_zonos_vector_dict.get(emotion, 0.0) for emotion in emotion_order]
                 
                 # --- Pass the Android 18 Tensor directly ---
                 cond_dict = make_cond_dict(
                     text=clean_text, 
                     emotion=emotion_list, 
                     language='en-us',
                     speaking_rate=speaking_rate, 
                     pitch_std=pitch_std,
                     speaker=speaker_embedding_tensor # Always Android 18
                 )

                 # --- Generation Logic ---
                 try:
                     conditioning = zonos_model.prepare_conditioning(cond_dict)
                     codes = zonos_model.generate(conditioning, disable_torch_compile=True)
                     wav_tensor = zonos_model.autoencoder.decode(codes).cpu()

                     if wav_tensor.numel() == 0:
                          continue
                     
                     audio_numpy = (wav_tensor.squeeze().numpy() * 32767).astype(np.int16) 
                     
                     output_filename = f"dialogue_{dialogue_index+1}_utterance_{utterance_index+1}.wav"
                     output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
                     
                     if audio_numpy.size == 0:
                          continue
                     write_wav(output_path, model_sampling_rate, audio_numpy)
                 except RuntimeError as e:
                     if "Attempting to deserialize object on a CUDA device" in str(e):
                         print(f"CUDA Error. Check GPU memory/setup. Error: {e}")
                     else:
                          print(f"Runtime Error: {e}")
                 except Exception as e:
                      print(f"General Error: {e}")

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON on line {dialogue_index+1}. Skipping dialogue.")
        except Exception as e:
            print(f"Unexpected error processing Dialogue {dialogue_index+1}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Optional: Clear cache if needed
    torch.cuda.empty_cache()
    synthesize_audio()
    print(f"\n--- Synthesis Complete! ---")
    print(f"All Android 18 audio files have been saved in '{OUTPUT_AUDIO_DIR}'.")