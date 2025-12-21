import torch
from transformers import pipeline
import json
import textwrap

print("🚀 Initializing the Multi-Turn LLM Therapist... This might take a moment.")
print("="*50)

# --- 1. โหลดโมเดล Llama 3 8B (เหมือนเดิม) ---
# โมเดลจะถูกโหลดแค่ครั้งเดียวตอนเริ่มสคริปต์
try:
    # FYI: The 'torch_dtype' is deprecated warning is normal and can be ignored.
    pipe = pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    print("✅ Llama 3 8B model loaded successfully onto the GPU!")
except Exception as e:
    print(f"❌ Failed to load the model. Error: {e}")
    exit()

# --- 2. โหลดข้อมูลจาก "สุดยอดคัมภีร์" (เหมือนเดิม) ---
json_path = '../merged_prompts_to_LLMs_therapist/final_llm_inputs.json'
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    print(f"✅ Successfully loaded {len(merged_data)} user data entries.")
except FileNotFoundError:
    print(f"❌ ERROR: Cannot find '{json_path}'.")
    exit()

# --- 3. เตรียม Prompt Template (ฉบับกระชับ) ---
prompt_template = """
SYSTEM PROMPT:
You are "Luna AI Therapist," an empathetic, warm, and observant AI assistant. You are non-judgmental, and your primary goal is to respond with compassion. You must utilize both the provided 'Transcript' of what the user said and the 'Vocal Context' describing how they said it. This multi-modal approach is crucial for understanding the user's underlying emotional state.

USER PROMPT:
Here is the data from a user session for you to analyze:

[Vocal Context]
{emotion_caption}

[Transcript]
"{transcript}"

[Your Task]
Based on all the provided information, craft a response as Luna AI Therapist. Your response must be brief and concise (around 2-3 sentences) to be easy for the user to read. It should still be warm and caring. Conclude with a single, gentle, open-ended question to encourage the user to elaborate.
"""
print("✅ Prompt template is ready.")
print("="*50)

# --- 4. (ส่วนใหม่!) กำหนดลำดับการสนทนาและเริ่มลูป! ---
# เราจะเรียงลำดับไฟล์เสียงจาก dialogue_1 เพื่อจำลองบทสนทนา
dialogue_sequence = [
    "dialogue_1_utterance_1.wav",
    "dialogue_1_utterance_2.wav",
    "dialogue_1_utterance_3.wav",
    "dialogue_1_utterance_4.wav",
    "dialogue_1_utterance_5.wav"
]

# สร้าง map เพื่อให้ค้นหาข้อมูลจากชื่อไฟล์ได้เร็วขึ้น
data_map = {item['file_name']: item for item in merged_data}

# เริ่มการจำลองบทสนทนา
for i, filename in enumerate(dialogue_sequence):
    print(f"\n\n--- TURN {i+1}/{len(dialogue_sequence)} ---")
    
    turn_data = data_map.get(filename)
    if not turn_data:
        print(f"⚠️  Warning: Data for {filename} not found. Skipping turn.")
        continue

    # แสดงข้อมูลฝั่ง Client
    print(f"CLIENT SAID: \"{turn_data['transcript']}\"")
    print(f"(Vocal Context: ...{turn_data['emotion_caption'][20:100]}...)") # แสดง Vocal Context แค่บางส่วน
    
    # สร้าง Prompt สำหรับเทิร์นนี้
    final_prompt = prompt_template.format(
        emotion_caption=turn_data['emotion_caption'],
        transcript=turn_data['transcript']
    )
    
    # ตั้งค่า Terminators สำหรับ Llama 3
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    # สั่งให้ AI ตอบกลับ
    outputs = pipe(
        final_prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # ตัดเอาเฉพาะส่วนที่เป็นคำตอบของ AI
    response_text = outputs[0]["generated_text"][len(final_prompt):]
    
    # แสดงผลลัพธ์ของ Therapist
    print("\nTHERAPIST'S RESPONSE:")
    wrapped_text = textwrap.fill(response_text.strip(), width=80)
    print(wrapped_text)
    print("="*50)

print("\n🎉 Multi-turn simulation complete!")