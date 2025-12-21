import json
import os
import sys
import torch
import torchaudio  # Library for audio file manipulation

# --- Initial Setup ---

# Add the core_zonos path so Python can find the Zonos library
# This script should be placed in the same directory as the core_zonos folder.
sys.path.append(os.path.join(os.getcwd(), 'core_zonos'))

# Import Zonos libraries after adding the path
try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
except ImportError as e:
    print("❌ Error: Could not import Zonos libraries.")
    print("   Please ensure this script is in the correct directory and all required libraries are installed.")
    print(f"   Error: {e}")
    sys.exit(1)

# Check if a GPU is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Using device: {DEVICE}")
if DEVICE == "cpu":
    print("⚠️ Warning: Running on CPU will be very slow!")

# --- Main Function ---

def generate_audio_from_descriptions(input_path, speaker_voice_path, zonos_params, limit_dialogues):
    """
    Function to read voice descriptions, generate audio with Zonos,
    and save the results and their parameters to folders.
    """
    print("🔥 Starting the final mission of The Forge: Generating real audio files!")

    # 1. Load the Zonos model
    print("    - Loading Zonos model... (This might take a moment)")
    try:
        model = Zonos.from_pretrained(
            # It is recommended to use the transformer model for the best quality.
            "core_zonos/models/Zonos-v0.1-transformer"
        ).to(DEVICE)
    except FileNotFoundError:
        print("❌ Error: Zonos model file not found!")
        print("    Please make sure you have downloaded the model and placed it in 'core_zonos/models/'.")
        sys.exit(1)

    # 2. Prepare the speaker voice for cloning
    print(f"    - Preparing speaker template from: {speaker_voice_path}")
    try:
        waveform, sample_rate = torchaudio.load(speaker_voice_path)
        speaker_embedding = model.make_speaker_embedding(wav=waveform, sr=sample_rate)
    except FileNotFoundError:
        print(f"❌ Error: Speaker template file not found at '{speaker_voice_path}'!")
        sys.exit(1)

    # 3. Create the main directories to store results
    audio_output_dir = 'generated_cbt_audio'
    params_output_dir = 'generated_cbt_params'  # <-- โฟลเดอร์ใหม่สำหรับเก็บพารามิเตอร์
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(params_output_dir, exist_ok=True)  # <-- สร้างโฟลเดอร์ใหม่
    print(f"    - Audio files will be saved to: '{audio_output_dir}'")
    print(f"    - Parameter files will be saved to: '{params_output_dir}'") # <-- ข้อความแจ้งเตือน

    # 4. Read the input file and start generating audio
    with open(input_path, 'r', encoding='utf-8') as infile:
        # อ่านไฟล์ทั้งหมดเก็บไว้ใน list ก่อน เพื่อจัดการ limit ได้ง่ายขึ้น
        dialogues = [line for line in infile if line.strip()]

        for i, line in enumerate(dialogues):
            if limit_dialogues and i >= limit_dialogues:
                print(f"\n⏹️ Reached the specified limit of {limit_dialogues} dialogues.")
                break
            
            try:
                data = json.loads(line)
                dialogue_id = data.get("dialogue_id", f"dialogue_{i+1:05d}")

                dialogue_audio_dir = os.path.join(audio_output_dir, dialogue_id)
                dialogue_params_dir = os.path.join(params_output_dir, dialogue_id) # <-- โฟลเดอร์ย่อยสำหรับพารามิเตอร์
                os.makedirs(dialogue_audio_dir, exist_ok=True)
                os.makedirs(dialogue_params_dir, exist_ok=True) # <-- สร้างโฟลเดอร์ย่อย

                print(f"\n--- Processing Dialogue ID: {dialogue_id} ---")

                voice_descriptions = data.get("voice_descriptions", [])
                
                if not voice_descriptions:
                    print("    - 🟡 No voice description in this dialogue, skipping...")
                    continue
                
                for j, desc_item in enumerate(voice_descriptions):
                    utterance_text = desc_item.get("client_utterance")
                    zonos_params_from_file = desc_item.get("voice_description_for_zonos", {})
                    
                    if not utterance_text:
                        continue
                    
                    # --- ส่วนของการสร้างเสียงและการบันทึกพารามิเตอร์ ---

                    # สร้าง cond_dict เหมือนเดิม
                    cond_dict = make_cond_dict(
                        text=utterance_text,
                        language="en-us",
                        speaker=speaker_embedding,
                        emotion=zonos_params_from_file.get("emotion_vector", [0.0]*7 + [1.0]),
                        pitch_std=zonos_params_from_file.get("pitch_std", 30.0),
                        speaking_rate=zonos_params_from_file.get("speaking_rate", 15.0),
                        device=DEVICE
                    )

                    # --- LUNA'S MLOPS UPGRADE ---
                    # ขั้นตอนใหม่: เตรียมและบันทึกพารามิเตอร์ลงไฟล์ JSON
                    params_to_save = {}
                    # แปลง Tensor กลับเป็น list/value ธรรมดาเพื่อให้ JSON บันทึกได้
                    for key, value in cond_dict.items():
                        if isinstance(value, torch.Tensor):
                            params_to_save[key] = value.cpu().tolist()
                        else:
                            params_to_save[key] = str(value) # แปลงเป็น string กันเหนียว
                    
                    params_to_save['text'] = utterance_text # เพิ่ม text เข้าไปเพื่อความชัดเจน

                    # สร้างชื่อไฟล์ .json ให้ตรงกับไฟล์ .wav
                    params_filename = f"utterance_{j+1:03d}.json"
                    params_filepath = os.path.join(dialogue_params_dir, params_filename)

                    # เขียนลงไฟล์ JSON
                    with open(params_filepath, 'w', encoding='utf-8') as f:
                        json.dump(params_to_save, f, ensure_ascii=False, indent=4)
                    print(f"     -> 📝 Params saved to: {params_filepath}")
                    # --- END OF UPGRADE ---
                    
                    # ส่วนการสร้างเสียงที่เหลือ ทำงานเหมือนเดิมทุกอย่าง
                    prefix_conditioning = model.prepare_conditioning(cond_dict=cond_dict)
                    
                    print(f"     - 🔊 Synthesizing audio for utterance {j+1}...") # แก้ไขให้ใช้ j
                    audio_codes = model.generate(prefix_conditioning=prefix_conditioning)
                    
                    output_waveform = model.autoencoder.decode(audio_codes)
                    
                    # Save the audio file
                    output_filename = f"utterance_{j+1:03d}.wav"
                    output_filepath = os.path.join(dialogue_audio_dir, output_filename)
                    
                    torchaudio.save(
                        output_filepath, 
                        output_waveform.reshape(1, -1).cpu().float(), 
                        model.autoencoder.sampling_rate
                    )
                    print(f"     -> ✅ Saved successfully to: {output_filepath}")

            except json.JSONDecodeError:
                print(f"    - ⚠️ Skipping non-JSON line: {line.strip()}")

    print("\n\n🎉🎉🎉 The Forge mission is complete! Our audio dataset has been successfully created! 🎉🎉🎉")

# --- ส่วนของการเรียกใช้งานฟังก์ชัน ---
# (สันนิษฐานว่าไดจิมีโค้ดส่วนนี้อยู่แล้ว)
if __name__ == '__main__':
    # ตั้งค่าพารามิเตอร์ต่างๆ ที่นี่
    INPUT_FILE = "datasets/cactus_voice_descriptions.jsonl"
    SPEAKER_FILE = "core_zonos/assets/exampleaudio.mp3"
    ZONOS_PARAMS = {} # อาจจะไม่จำเป็นแล้ว เพราะอ่านจากไฟล์
    DIALOGUE_LIMIT = 2 # ตั้งค่า limit ที่นี่

    generate_audio_from_descriptions(
        input_path=INPUT_FILE,
        speaker_voice_path=SPEAKER_FILE,
        zonos_params=ZONOS_PARAMS,
        limit_dialogues=DIALOGUE_LIMIT
    )
