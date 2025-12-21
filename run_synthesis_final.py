import torch
import torchaudio # <--- Added import torchaudio
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
INPUT_JSONL_FILE = "datasets/final_hierarchical_output_v_openai.jsonl" 
MODEL_ID = "Zonos-v0.1-transformer" 
ZONOS_MODEL_DIR = os.path.join("core_zonos", "models", MODEL_ID)
OUTPUT_AUDIO_DIR = "final_audio_output_final" 

# --- UPDATED: Paths to your sample audio files ---
# *** Paths relative to the script location (test directory) ***
MALE_VOICE_SAMPLE_PATH = os.path.join("core_zonos", "assets", "male_voice_android17.wav")   # <--- Updated!
FEMALE_VOICE_SAMPLE_PATH = os.path.join("core_zonos", "assets", "female_voice_android18.wav") # <--- Updated!
# --- END UPDATED ---

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
    print("Creating speaker embeddings from sample audio files...")
    male_embedding_tensor: Optional[torch.Tensor] = None
    female_embedding_tensor: Optional[torch.Tensor] = None

    # Determine the model's expected sampling rate
    model_sampling_rate = int(zonos_model.autoencoder.sampling_rate)
    print(f"Model expects audio at {model_sampling_rate} Hz.")

    try:
        wav_male, sr_male = torchaudio.load(MALE_VOICE_SAMPLE_PATH)
        # Resample if necessary
        if sr_male != model_sampling_rate:
            print(f"Resampling male audio from {sr_male} Hz to {model_sampling_rate} Hz...")
            wav_male = torchaudio.functional.resample(wav_male, sr_male, model_sampling_rate)
            sr_male = model_sampling_rate # Update sample rate after resampling
        male_embedding_tensor = zonos_model.make_speaker_embedding(wav_male.to(device), sr_male) # Use the (potentially updated) sr_male
        print(f"Successfully created male speaker embedding from '{MALE_VOICE_SAMPLE_PATH}'")
    except FileNotFoundError:
        print(f"Warning: Male voice sample file not found at '{MALE_VOICE_SAMPLE_PATH}'. Will use default/random speaker for male.")
    except Exception as e:
        print(f"Error creating male embedding: {e}")

    try:
        wav_female, sr_female = torchaudio.load(FEMALE_VOICE_SAMPLE_PATH)
        # Resample if necessary
        if sr_female != model_sampling_rate:
            print(f"Resampling female audio from {sr_female} Hz to {model_sampling_rate} Hz...")
            wav_female = torchaudio.functional.resample(wav_female, sr_female, model_sampling_rate)
            sr_female = model_sampling_rate # Update sample rate after resampling
        female_embedding_tensor = zonos_model.make_speaker_embedding(wav_female.to(device), sr_female) # Use the (potentially updated) sr_female
        print(f"Successfully created female speaker embedding from '{FEMALE_VOICE_SAMPLE_PATH}'")
    except FileNotFoundError:
        print(f"Warning: Female voice sample file not found at '{FEMALE_VOICE_SAMPLE_PATH}'. Will use default/random speaker for female.")
    except Exception as e:
        print(f"Error creating female embedding: {e}")
    
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
            
            # --- Extract Speaker Gender ---
            intake_form_str = data.get("original_dialogue_full", {}).get("intake_form", "")
            dialogue_speaker_gender = None
            gender_match = re.search(r"Gender:\s*(\w+)", intake_form_str, re.IGNORECASE)
            if gender_match:
                dialogue_speaker_gender = gender_match.group(1).lower() 
            
            # --- Select the correct embedding tensor ---
            selected_speaker_tensor: Optional[torch.Tensor] = None
            if dialogue_speaker_gender == 'male':
                selected_speaker_tensor = male_embedding_tensor
            elif dialogue_speaker_gender == 'female':
                selected_speaker_tensor = female_embedding_tensor
            
            # --- Logging for speaker selection ---
            if not dialogue_speaker_gender:
                 if dialogue_index < 5: # Log only for the first few dialogues to avoid spam
                    print(f"\nInfo: Using default/random speaker for Dialogue {dialogue_index+1} (gender not found in intake form).")
            elif selected_speaker_tensor is None:
                 if dialogue_index < 5:
                    print(f"\nInfo: Using default/random speaker for Dialogue {dialogue_index+1} (gender: {dialogue_speaker_gender}) due to missing/failed embedding creation.")
            # --- End Logging ---

            directed_utterances = data.get("llm_output", {}).get("directed_utterances", [])
            # Handle potential error in llm_output
            if isinstance(directed_utterances, str) and "error" in directed_utterances.lower():
                 print(f"Warning: Skipping Dialogue {dialogue_index+1} due to error in 'llm_output': {data.get('llm_output')}")
                 continue
            if not isinstance(directed_utterances, list):
                 print(f"Warning: Skipping Dialogue {dialogue_index+1} due to unexpected format in 'directed_utterances'.")
                 continue


            # --- Process Each Utterance ---
            for utterance_index, utterance_data in enumerate(tqdm(directed_utterances, desc=f"  Dialogue {dialogue_index+1}", leave=False)):
                 # --- NEW: Added type check for utterance_data ---
                 if not isinstance(utterance_data, dict):
                     print(f"Warning: Skipping invalid utterance data in Dialogue {dialogue_index+1}, Utterance {utterance_index+1}: {utterance_data}")
                     continue
                 # --- END NEW ---

                 utterance_text_with_tags = utterance_data.get("utterance_text", "")
                 final_zonos_vector_dict = utterance_data.get("final_zonos_vector", {})
                 gen_params = utterance_data.get("generation_parameters", {})
                 speaking_rate = gen_params.get("speaking_rate")
                 pitch_std = gen_params.get("pitch_std")
                 
                 # --- NEW: Check if final_zonos_vector is valid dict ---
                 if not isinstance(final_zonos_vector_dict, dict):
                      print(f"Warning: Skipping utterance in Dialogue {dialogue_index+1}, Utterance {utterance_index+1} due to invalid 'final_zonos_vector': {final_zonos_vector_dict}")
                      continue
                 # --- END NEW ---
                 
                 clean_text = re.sub(r'\[.*?\]', '', utterance_text_with_tags).strip()
                 
                 # Skip if text is empty OR if the vector dict is empty after potential error correction
                 if not clean_text or not final_zonos_vector_dict: 
                     # print(f"Skipping Dialogue {dialogue_index+1}, Utterance {utterance_index+1} due to empty text or vector.") # Optional debug
                     continue 
                     
                 emotion_list = [final_zonos_vector_dict.get(emotion, 0.0) for emotion in emotion_order]
                 
                 # --- Pass the SELECTED SPEAKER TENSOR (or None) ---
                 cond_dict = make_cond_dict(
                     text=clean_text, 
                     emotion=emotion_list, 
                     language='en-us',
                     speaking_rate=speaking_rate, 
                     pitch_std=pitch_std,
                     speaker=selected_speaker_tensor 
                 )

                 # --- Generation Logic ---
                 try:
                     conditioning = zonos_model.prepare_conditioning(cond_dict)
                     codes = zonos_model.generate(conditioning, disable_torch_compile=True)
                     wav_tensor = zonos_model.autoencoder.decode(codes).cpu()

                     if wav_tensor.numel() == 0:
                          print(f"Warning: Generated empty audio for Dialogue {dialogue_index+1}, Utterance {utterance_index+1}. Skipping.")
                          continue
                     
                     audio_numpy = (wav_tensor.squeeze().numpy() * 32767).astype(np.int16) 
                     
                     output_filename = f"dialogue_{dialogue_index+1}_utterance_{utterance_index+1}.wav"
                     output_path = os.path.join(OUTPUT_AUDIO_DIR, output_filename)
                     # Check if numpy array is empty before writing
                     if audio_numpy.size == 0:
                          print(f"Warning: Numpy array is empty for Dialogue {dialogue_index+1}, Utterance {utterance_index+1}. Skipping write.")
                          continue
                     write_wav(output_path, model_sampling_rate, audio_numpy) # Use model_sampling_rate
                 except RuntimeError as e:
                     if "Attempting to deserialize object on a CUDA device" in str(e):
                         print(f"CUDA Error during generation for Dialogue {dialogue_index+1}, Utterance {utterance_index+1}. Check GPU memory/setup. Skipping utterance. Error: {e}")
                     else:
                          print(f"Runtime Error during generation/saving for Dialogue {dialogue_index+1}, Utterance {utterance_index+1}: {e}")
                 except Exception as e:
                      print(f"General Error during generation/saving for Dialogue {dialogue_index+1}, Utterance {utterance_index+1}: {e}")

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON on line {dialogue_index+1}. Skipping dialogue.")
        except Exception as e:
            print(f"Unexpected error processing Dialogue {dialogue_index+1}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    synthesize_audio()
    print(f"\n--- Synthesis Complete! ---")
    print(f"All audio files have been saved in the '{OUTPUT_AUDIO_DIR}' directory.")