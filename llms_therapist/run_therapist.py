import torch
from transformers import pipeline
import json
import textwrap

print("🚀 Initializing the LLM Therapist... Please wait.")
print("="*50)

# --- 1. โหลดโมเดล Llama 3 8B (The A.I. Core) ---
# นี่คือส่วนที่เรียกใช้ "มันสมอง" ของ Therapist ของเราค่ะ
# device_map="auto" จะสั่งให้ accelerate จัดการส่งโมเดลไปที่ GPU ให้เองเลย
try:
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

# --- 2. โหลดข้อมูลจาก "สุดยอดคัมภีร์" (final_llm_inputs.json) ---
# เราจะไปดึงข้อมูลที่ผ่านการ "ฟิวชั่น" มาจาก Directory ที่แล้ว
json_path = '../merged_prompts_to_LLMs_therapist/final_llm_inputs.json'
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    print(f"✅ Successfully loaded {len(merged_data)} user data entries.")
except FileNotFoundError:
    print(f"❌ ERROR: Cannot find '{json_path}'.")
    print("   Please make sure the 'merged_prompts_to_LLMs_therapist' directory is in the correct place and contains the JSON file.")
    exit()

# --- 3. เลือกข้อมูลมาทดสอบ และสร้าง "คาถาอัญเชิญ" (Prompt) ---
# เราจะลองทดสอบกับข้อมูลชุดแรก (index 0) ก่อนนะคะ
test_data = merged_data[0] 

# นี่คือ Prompt Template ภาษาอังกฤษที่เราออกแบบไว้
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
Based on all the provided information, craft a response as Luna AI Therapist. Your response should be warm, caring, and empathetic. Conclude your response with a gentle, open-ended question to encourage the user to elaborate on their feelings. And limited the response for each turn to not have anymore than 3 sentenses. 
"""

# นำข้อมูลมาใส่ใน Template
final_prompt = prompt_template.format(
    emotion_caption=test_data['emotion_caption'],
    transcript=test_data['transcript']
)

print("✅ Prompt created successfully.")
print("="*50)


# --- 4. สั่งให้ LLM Therapist ทำงาน! (The Activation) ---
print("🧠 Sending prompt to the Therapist... Awaiting response...")
print("\n--- INPUT PROMPT ---")
print(final_prompt)
print("--------------------")

# ส่ง Prompt เข้าไปใน pipeline และตั้งค่าการสร้างข้อความ
# เราใช้ terminators เพื่อบอก Llama 3 ว่าให้จบประโยคตรงไหน
terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipe(
    final_prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

# --- 5. แสดงผลลัพธ์ที่ได้ ---
# ผลลัพธ์ที่ได้จะรวม Prompt ของเราเข้าไปด้วย เราต้องตัดมันออกเพื่อเอาเฉพาะคำตอบของ AI
response_text = outputs[0]["generated_text"][len(final_prompt):]

print("\n\n--- THERAPIST'S RESPONSE ---")
# ใช้ textwrap เพื่อให้ข้อความยาวๆ ขึ้นบรรทัดใหม่สวยงาม
wrapped_text = textwrap.fill(response_text, width=80)
print(wrapped_text)
print("="*50)
print("🎉 Mission Complete!")