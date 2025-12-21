import json
import os

print("Starting the fusion process... 🐉")

# --- 1. ระบุตำแหน่งของ "ม้วนคัมภีร์" ทั้งสอง ---
# ใช้ ../ เพื่อถอยกลับไปหนึ่ง Directory
emotion_file_path = '../se_c/emotion_results.json'
transcription_file_path = '../auto_speech_recog/transcription_results.json'

# --- 2. อ่านข้อมูลจาก "ม้วนคัมภีร์" ทั้งสอง ---
try:
    with open(emotion_file_path, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)
    print(f"✅ Successfully loaded {len(emotion_data)} emotion captions.")

    with open(transcription_file_path, 'r', encoding='utf-8') as f:
        transcription_data = json.load(f)
    print(f"✅ Successfully loaded {len(transcription_data)} transcripts.")
except FileNotFoundError as e:
    print(f"❌ ERROR: Cannot find the file! Make sure you run the other scripts first.")
    print(e)
    exit()

# --- 3. สร้าง "แผนที่" เพื่อให้จับคู่ได้เร็วขึ้น ---
# เปลี่ยน list ของ transcript ให้เป็น dictionary เพื่อให้ค้นหาด้วยชื่อไฟล์ได้ง่าย
transcripts_map = {item['file']: item['transcript'].strip() for item in transcription_data}
print("🗺️ Created a map for quick lookups.")

# --- 4. เริ่มทำการ "ฟิวชั่น"! ---
merged_data = []
for emotion_item in emotion_data:
    file_name = emotion_item['file']
    
    # ค้นหา transcript ที่คู่กันจาก "แผนที่" ของเรา
    transcript_text = transcripts_map.get(file_name, "--- TRANSCRIPT NOT FOUND ---")
    
    # รวมร่างข้อมูล!
    merged_item = {
        "file_name": file_name,
        "transcript": transcript_text,
        "emotion_caption": emotion_item['caption']
    }
    merged_data.append(merged_item)

print(f"✨ Fusion complete! {len(merged_data)} items have been merged.")

# --- 5. บันทึกผลลัพธ์สุดท้ายลง "สุดยอดคัมภีร์" ---
final_output_filename = 'final_llm_inputs.json'
with open(final_output_filename, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"🎉 Success! Final combined data saved to '{final_output_filename}'")

# --- แสดงตัวอย่างผลลัพธ์ 2-3 อันแรก ---
print("\n--- Here is a sample of the merged data: ---")
for item in merged_data[:3]:
    print(json.dumps(item, indent=2, ensure_ascii=False))
    print("-" * 20)