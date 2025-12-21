import json
from datasets import load_dataset

# --- Configuration ---
DATASET_ID = "LangAGI-Lab/cactus"
SPLIT = "train"
OUTPUT_FILENAME = "cactus_huggingface_train.jsonl"

print(f"Starting download for dataset: {DATASET_ID} (split: {SPLIT})")

try:
    # 1. โหลด dataset จาก Hugging Face Hub
    dataset = load_dataset(DATASET_ID, split=SPLIT)
    
    print("Dataset loaded. Now writing to file...")
    
    # 2. เขียนข้อมูลลงไฟล์ .jsonl โดยใช้ json.dumps (วิธีที่ถูกต้อง)
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        for item in dataset:
            # json.dumps จะแปลง Python dict ให้เป็น JSON string ที่ถูกต้องเสมอ
            f.write(json.dumps(item) + '\n')

    record_count = len(dataset)
    print("\n--- Download and Save Complete! ---")
    print(f"Successfully saved {record_count} records to {OUTPUT_FILENAME}")

except Exception as e:
    print(f"An error occurred: {e}")