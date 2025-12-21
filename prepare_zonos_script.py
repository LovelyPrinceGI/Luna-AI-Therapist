import json
import os

# --- 1. Configuration ---
INPUT_FILE = "datasets/final_hierarchical_output_v_openai.jsonl" # ไฟล์ผลลัพธ์สุดท้ายของเรา
OUTPUT_DIR = "zonos_scripts" # โฟลเดอร์สำหรับเก็บไฟล์ที่จะป้อนให้ Zonos

# --- 2. Main Logic ---
def create_zonos_scripts():
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: '{OUTPUT_DIR}'")

    print(f"Reading from '{INPUT_FILE}'...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                try:
                    # โหลดข้อมูลจากแต่ละบรรทัด
                    data = json.loads(line.strip())
                    
                    llm_output = data.get("llm_output", {})
                    directed_utterances = llm_output.get("directed_utterances", [])
                    
                    full_zonos_script = []
                    
                    # วนลูปไปในแต่ละ utterance ของบทสนทนานั้นๆ
                    for utterance_data in directed_utterances:
                        text = utterance_data.get("utterance_text", "")
                        vector = utterance_data.get("final_zonos_vector", {})
                        
                        # --- NEW: ดึง generation_parameters ออกมา ---
                        gen_params = utterance_data.get("generation_parameters", {})
                        speaking_rate = gen_params.get("speaking_rate")
                        pitch_std = gen_params.get("pitch_std")
                        # --- END NEW ---
                        
                        if not text or not vector:
                            continue # ข้ามถ้าข้อมูลไม่สมบูรณ์
                        
                        # สร้าง string ของ emotion attributes
                        attributes_list = []
                        for emotion, value in vector.items():
                            # Zonos อาจจะต้องการชื่อ emotion เป็นตัวเล็ก
                            attributes_list.append(f'{emotion.lower()}="{value}"')
                        
                        # --- NEW: เพิ่ม generation parameters เข้าไปใน list ---
                        if speaking_rate is not None:
                            # ชื่อ attribute อาจจะต้องเป็น 'rate' หรือ 'speaking_rate' 
                            # ขึ้นอยู่กับ Zonos นะคะ ไดจิต้องเช็คดูอีกทีค่ะ
                            attributes_list.append(f'speaking_rate="{speaking_rate}"')
                        
                        if pitch_std is not None:
                            # ชื่อ attribute อาจจะต้องเป็น 'pitch' หรือ 'pitch_std'
                            attributes_list.append(f'pitch_std="{pitch_std}"')
                        # --- END NEW ---

                        attributes_str = " ".join(attributes_list)
                        
                        # สร้าง tag <emotion> ที่สมบูรณ์
                        # ตอนนี้มันจะมีหน้าตาประมาณ <emotion happiness="0.8" speaking_rate="1.1">Hello</emotion>
                        formatted_line = f'<emotion {attributes_str}>{text}</emotion>'
                        full_zonos_script.append(formatted_line)
                    
                    # ถ้ามีข้อมูลที่แปลงแล้ว ก็ให้บันทึกเป็นไฟล์
                    if full_zonos_script:
                        output_filename = os.path.join(OUTPUT_DIR, f"dialogue_{i+1}_zonos.txt")
                        with open(output_filename, 'w', encoding='utf-8') as outfile:
                            # รวมทุก utterance เข้าด้วยกัน คั่นด้วย new line
                            outfile.write("\n".join(full_zonos_script))
                        # print(f"Successfully created '{output_filename}'") # อาจจะปิดไปก่อนถ้ามัน log เยอะไป

                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON on line {i+1}. Skipping.")
        
        print(f"Done! All Zonos scripts are created in '{OUTPUT_DIR}'.")

    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

if __name__ == "__main__":
    create_zonos_scripts()