from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import os
import glob
from tqdm import tqdm # เพิ่ม tqdm เพื่อทำ progress bar สวยๆ

# --- 1. Load the Model (เหมือนเดิม) ---
print("Loading Whisper model...")
processor = WhisperProcessor.from_pretrained("cahya/whisper-large-audio-captioning-v1.0")
model = WhisperForConditionalGeneration.from_pretrained("cahya/whisper-large-audio-captioning-v1.0")
print("Model loaded successfully!")


# --- 2. Find All Audio Files (ส่วนอัปเกรด) ---
# ระบุที่อยู่ของ 'ห้องอัดเสียง' ของเรา
audio_folder_path = "../final_audio_output_final/"

# ใช้ glob เพื่อสร้าง 'เพลย์ลิสต์' ของไฟล์ .wav ทั้งหมด
# os.path.join() จะช่วยเชื่อมต่อ path ให้ถูกต้อง
wav_files = glob.glob(os.path.join(audio_folder_path, "*.wav"))
print(f"Found {len(wav_files)} audio files to process in '{audio_folder_path}'")


# --- 3. Process Each File in a Loop (ส่วนอัปเกรด) ---
# สร้าง list ว่างๆ เพื่อเก็บผลลัพธ์
results = []

# ใช้ tqdm ครอบเพลย์ลิสต์ของเราเพื่อแสดง progress bar
for audio_path in tqdm(wav_files, desc="Transcribing audio files"):
    try:
        # โหลดไฟล์เสียงจาก 'เพลย์ลิสต์' ของเรา
        audio, sr = librosa.load(audio_path, sr=16000)

        # ประมวลผลและถอดเสียง (เหมือนเดิม)
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # เก็บชื่อไฟล์และผลลัพธ์
        file_name = os.path.basename(audio_path)
        results.append({"file": file_name, "caption": transcription})

    except Exception as e:
        # กรณีที่ไฟล์เสียงบางไฟล์อาจมีปัญหา
        file_name = os.path.basename(audio_path)
        results.append({"file": file_name, "caption": f"ERROR: {e}"})

# --- 4. Print All Results (ส่วนอัปเกรด) ---
print("\n--- Transcription Complete! ---\n")
for result in results:
    print(f"File: {result['file']} -> Caption: {result['caption']}\n")

# --- 5. บันทึกผลลัพธ์ลงม้วนคัมภีร์! (ส่วนที่เพิ่มใหม่) ---
output_filename = "emotion_results.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ Results successfully saved to '{output_filename}'")