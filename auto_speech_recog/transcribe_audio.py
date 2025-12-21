import torch
import glob
import os
from tqdm import tqdm
from transformers import pipeline
import json

# --- 1. เตรียม "ไม้กายสิทธิ์" (Load the Model using Pipeline) ---
# ตรวจสอบว่ามี GPU (NVIDIA CUDA) ให้ใช้ไหม ถ้ามีจะเร็วขึ้นมาก!
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model 'openai/whisper-large-v3' onto device: {device}")

# สร้าง pipeline เหมือนเรียกใช้เวทมนตร์อัตโนมัติ
# มันจะจัดการเรื่อง processor และ model ให้เราเองเลยค่ะ
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch_dtype,
    device=device,
)

print("Model loaded successfully!")


# --- 2. หา "วัตถุดิบ" ทั้งหมด (Find All Audio Files) ---
# Path ไปยังโฟลเดอร์ที่เก็บไฟล์เสียงของเรา (ใช้ ../ เพื่อถอยกลับไปหนึ่งขั้น)
audio_folder_path = "../final_audio_output_final/"
wav_files = glob.glob(os.path.join(audio_folder_path, "*.wav"))
print(f"Found {len(wav_files)} audio files to process in '{audio_folder_path}'")


# --- 3. เริ่ม "ร่ายเวทมนตร์" ทีละไฟล์ (Process Each File) ---
results = []
# ใช้ tqdm เพื่อให้เห็น progress bar สวยๆ ตอนทำงาน
for audio_path in tqdm(wav_files, desc="Transcribing audio files"):
    try:
        # ส่ง path ของไฟล์เสียงให้ pipeline จัดการได้เลย!
        transcription_result = pipe(audio_path)
        
        # ผลลัพธ์ที่ได้จะเป็น Dictionary หน้าตาประมาณ {'text': 'สวัสดีครับ...'}
        transcript_text = transcription_result["text"]

        # เก็บชื่อไฟล์และข้อความที่ถอดเสียงได้
        file_name = os.path.basename(audio_path)
        results.append({"file": file_name, "transcript": transcript_text})

    except Exception as e:
        file_name = os.path.basename(audio_path)
        results.append({"file": file_name, "transcript": f"ERROR: {e}"})


# --- 4. แสดงผลลัพธ์ทั้งหมด (ส่วนอัปเกรด) ---
print("\n--- Transcription Complete! ---\n")
for result in results:
    # เพิ่ม .strip() เพื่อตัดช่องว่างที่ไม่จำเป็นหน้า-หลังข้อความออกค่ะ
    print(f"File: {result['file']} -> Transcript: {result['transcript'].strip()}\n")

# --- 5. บันทึกผลลัพธ์ลงม้วนคัมภีร์! (ส่วนที่เพิ่มใหม่) ---
output_filename = "transcription_results.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"✅ Results successfully saved to '{output_filename}'")