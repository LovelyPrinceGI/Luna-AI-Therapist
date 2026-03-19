# ... (Imports and Configuration including MIN/MAX values unchanged) ...
import torch
from transformers import pipeline
import json
import time
from tqdm import tqdm
import pandas as pd
import re
from collections import defaultdict
import os

# --- 1. Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
UTTERANCE_RULES_FILE = "utterance_level_rules.csv"
FRAME_RULES_FILE = "frame_level_rules.csv"
INPUT_FILE = "cactus_huggingface_train.jsonl"
OUTPUT_FILE = "final_hierarchical_output_v13_d20.jsonl" # <<< ตรวจสอบชื่อไฟล์ Output ที่ต้องการ
MAX_NEW_TOKENS = 2048 # <<< เพิ่ม Token ตามที่คุยกัน
NUM_SAMPLES_TO_PROCESS = 20
MAX_RETRIES = 10 # <<< ลด Retry กลับมาเป็น 3 หรือตามต้องการ

# Define min/max for Zonos parameters
MIN_SPEAKING_RATE = 10.0
MAX_SPEAKING_RATE = 25.0
MIN_PITCH_STD = 20.0
MAX_PITCH_STD = 150.0

# --- 2. Load Model ---
# ... (unchanged) ...
print(f"Loading model: {MODEL_ID}...")
# Make sure to include device_map="auto" to use available GPUs
pipe = pipeline("text-generation", model=MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
print("Model loaded successfully!")


# --- 3. Load and Format Rules from CSV Files ---
# <<< MODIFIED load_rules to be more robust with JSON parsing >>>
def load_rules():
    """Loads rules from CSVs into DataFrames and formatted strings."""
    print("Loading rules from CSV files...")
    try:
        # <<< ใช้ dtype=str เพื่อป้องกัน Pandas แปลงค่าผิดประเภท >>>
        utterance_df = pd.read_csv(UTTERANCE_RULES_FILE, dtype=str).fillna('')
        frame_df = pd.read_csv(FRAME_RULES_FILE, dtype=str).fillna('')

        # --- Format utterance rules with FULL details ---
        utterance_rules_list = []
        for index, row in utterance_df.iterrows():
            key = str(row.get('stage_direction_key','')).strip()
            if not key:
                print(f"Warning: Skipping empty stage_direction_key at index {index} in {UTTERANCE_RULES_FILE}")
                continue

            vector_str_raw = str(row.get('primary_zonos_vector_value', '{}')).strip()
            speaking_rate_str = str(row.get('speaking_rate', '15.0')).strip()
            pitch_std_str = str(row.get('pitch_std', '20.0')).strip()
            zonos_vector = {}
            speaking_rate = 15.0
            pitch_std = 20.0

            # Attempt to parse Zonos vector
            try:
                # Find the JSON part more reliably
                json_match = re.search(r'\{.*\}', vector_str_raw)
                if json_match:
                    json_str_cleaned = json_match.group(0).replace("'", '"') # Replace single quotes
                    # <<< เพิ่มการจัดการกับ double double quotes ถ้ามี เช่น ""Happiness"" >>>
                    json_str_cleaned = re.sub(r'""(\w+)""', r'"\1"', json_str_cleaned)
                    zonos_vector = json.loads(json_str_cleaned)
                else:
                    print(f"Warning: Could not extract dict structure for key '{key}' from '{vector_str_raw}'. Using empty vector.")
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"Warning: JSON parse error for key '{key}' from '{vector_str_raw}'. Error: {e}. Using empty vector.")

            # Attempt to parse speaking rate
            try:
                speaking_rate = float(speaking_rate_str)
            except (ValueError, TypeError):
                print(f"Warning: Invalid speaking_rate '{speaking_rate_str}' for key '{key}'. Using default 15.0.")

            # Attempt to parse pitch std
            try:
                pitch_std = float(pitch_std_str)
            except (ValueError, TypeError):
                print(f"Warning: Invalid pitch_std '{pitch_std_str}' for key '{key}'. Using default 20.0.")


            # Create the full definition dictionary using parsed values
            full_definition = {
                "primary_zonos_vector_value": zonos_vector, # Use parsed dict
                "speaking_rate": speaking_rate,             # Use parsed float
                "pitch_std": pitch_std                      # Use parsed float
            }
            # Convert the full definition back to a compact JSON string for the prompt
            try:
                 rule_json_string = json.dumps(full_definition)
                 utterance_rules_list.append(f"- `{key}`: {rule_json_string}")
            except TypeError as e:
                 print(f"Error creating JSON string for prompt for key '{key}'. Error: {e}")


        utterance_rules_str = "\n".join(utterance_rules_list)
        # --- END OF UPDATE ---

        # Format frame rules for the prompt (can remain simple but use json.dumps for safety)
        frame_rules_list = []
        for index, row in frame_df.iterrows():
            key = str(row.get('frame_level_key','')).strip()
            if not key:
                 print(f"Warning: Skipping empty frame_level_key at index {index} in {FRAME_RULES_FILE}")
                 continue
            value_raw = str(row.get('zonos_vector_value','{}')).strip()
            # Try to represent it cleanly for the prompt
            try:
                 # Attempt to parse and dump cleanly if it's JSON-like
                 json_match = re.search(r'\{.*\}', value_raw)
                 if json_match:
                      json_str_cleaned = json_match.group(0).replace("'",'"')
                      json_str_cleaned = re.sub(r'""(\w+)""', r'"\1"', json_str_cleaned)
                      parsed_value = json.loads(json_str_cleaned)
                      frame_rules_list.append(f"- `{key}`: {json.dumps(parsed_value)}")
                 else:
                      # If not JSON-like, dump as string
                      frame_rules_list.append(f"- `{key}`: {json.dumps(value_raw)}")
            except (json.JSONDecodeError, AttributeError):
                 # Fallback: dump raw string if parsing fails
                 frame_rules_list.append(f"- `{key}`: {json.dumps(value_raw)}")


        frame_rules_str = "\n".join(frame_rules_list)

        print("Rules loaded and formatted successfully.")
        # <<< คืนค่า DataFrame ที่อ่านแบบ String ไปก่อน >>>
        return utterance_df, frame_df, utterance_rules_str, frame_rules_str

    except FileNotFoundError as e:
        print(f"Error: Could not find rule file: {e}. Please ensure CSV files are in the same directory.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during rule loading: {e}")
        # import traceback # Uncomment for debugging
        # traceback.print_exc() # Uncomment for debugging
        exit()

# --- Automatic Rule Updater ---
# ... (update_rule_file function unchanged, assumes clamped values) ...
def update_rule_file(file_path, df, new_rule_data, rule_type):
    """
    Appends a new rule to the specified CSV file and updates the in-memory DataFrame.
    (Assumes values in new_rule_data['definition'] are already clamped/validated)
    """
    try:
        print(f"\n✨ New {'utterance' if rule_type == 'utterance' else 'frame'} rule! Updating {file_path}...")

        key = new_rule_data.get('key')
        definition = new_rule_data.get('definition', {})
        if not key:
             print("❌ Error: Cannot update rule file, key is missing.")
             return df

        if rule_type == 'utterance':
            # Ensure zonos_vector is a string for CSV
            zonos_vector_csv_str = json.dumps(definition.get('primary_zonos_vector_value', definition.get('zonos_vector', {})))

            new_row_data = {
                'stage_direction_key': key,
                'primary_zonos_vector_value': zonos_vector_csv_str,
                'speaking_rate': definition.get('speaking_rate', 15.0), # Use default if missing
                'pitch_std': definition.get('pitch_std', 20.0) # Use default if missing
            }
            new_row = pd.DataFrame([new_row_data])
            # <<< เขียน Header ถ้าไฟล์ไม่มีอยู่ หรือไฟล์ว่างเปล่า >>>
            new_row.to_csv(file_path, mode='a', header=not os.path.exists(file_path) or os.path.getsize(file_path) == 0, index=False)

        elif rule_type == 'frame':
             # Ensure zonos_vector_value is a string for CSV
            zonos_vector_value_csv_str = json.dumps(definition.get('zonos_vector_value', definition.get('zonos_vector', {})))
            new_row_data = {
                'frame_level_key': key,
                'zonos_vector_value': zonos_vector_value_csv_str
            }
            new_row = pd.DataFrame([new_row_data])
            new_row.to_csv(file_path, mode='a', header=not os.path.exists(file_path) or os.path.getsize(file_path) == 0, index=False)

        # Update the in-memory DataFrame (Important: Convert types for consistency if needed later)
        # Check if the key already exists before concatenating
        key_column = 'stage_direction_key' if rule_type == 'utterance' else 'frame_level_key'
        # <<< ตรวจสอบค่าซ้ำให้ดีขึ้น >>>
        if key in df[key_column].astype(str).values:
             print(f"ℹ️ Rule '{key}' already exists in DataFrame. Skipping concat.")
             updated_df = df
        else:
             # <<< ทำให้แน่ใจว่า new_row มี column ตรงกับ df ก่อน concat >>>
             new_row_aligned = pd.DataFrame([new_row_data], columns=df.columns)
             updated_df = pd.concat([df, new_row_aligned], ignore_index=True)
             print(f"✅ Successfully added new rule to CSV and DataFrame: {key}")

        return updated_df

    except Exception as e:
        print(f"❌ Error updating rule file {file_path}: {e}")
        # import traceback # Uncomment for debugging
        # traceback.print_exc() # Uncomment for debugging
        # print("Debug - Data that caused error:", new_rule_data) # Uncomment for debugging
        return df # Return original dataframe on failure


# --- 4. Vector Calculation Function (Luna's Weighted Fusion Formula) ---
# <<< MODIFIED calculate_final_vector for robustness >>>
def calculate_final_vector(utterance_output, utterance_df, frame_df, w_utterance=1.0, w_frame=0.25):
    """
    Calculates the final Zonos vector using "Luna's Weighted Fusion Formula."
    Handles both existing rules from dataframes and potentially new rules defined in the utterance_output.
    Improved robustness for parsing vectors.
    """
    try:
        final_vector = defaultdict(float)
        base_vector = {}

        # --- Step 1: Get the Utterance Level Vector ---
        utterance_key = utterance_output.get('utterance_level_direction')

        # Priority 1: Use new rule definition if present
        if utterance_output.get('is_new_utterance_rule') and isinstance(utterance_output.get('new_utterance_rule_definition'), dict):
            new_def = utterance_output['new_utterance_rule_definition']
            base_vector = new_def.get('primary_zonos_vector_value', new_def.get('zonos_vector', {}))
            if not isinstance(base_vector, dict): base_vector = {} # Ensure it's a dict

        # Priority 2: Use existing rule definition if fetched earlier
        elif 'existing_rule_definition' in utterance_output and isinstance(utterance_output['existing_rule_definition'], dict):
             existing_def = utterance_output['existing_rule_definition']
             base_vector = existing_def.get('primary_zonos_vector_value', {})
             if not isinstance(base_vector, dict): base_vector = {} # Ensure it's a dict

        # Priority 3: Look up in DataFrame as a fallback (less reliable due to potential parsing issues here)
        elif utterance_key:
            # <<< ค้นหาใน DataFrame ที่อ่านค่าเป็น String >>>
            utterance_rule = utterance_df[utterance_df['stage_direction_key'].astype(str) == str(utterance_key)]
            if not utterance_rule.empty:
                base_vector_str_raw = utterance_rule.iloc[0].get('primary_zonos_vector_value', '{}')
                try:
                    # <<< ใช้ Logic การ Parse ที่ปรับปรุงแล้วเหมือนใน load_rules >>>
                    json_match = re.search(r'\{.*\}', str(base_vector_str_raw))
                    if json_match:
                         json_str_cleaned = json_match.group(0).replace("'", '"')
                         json_str_cleaned = re.sub(r'""(\w+)""', r'"\1"', json_str_cleaned)
                         base_vector = json.loads(json_str_cleaned)
                    else: base_vector = {}
                except (json.JSONDecodeError, AttributeError, TypeError):
                    print(f"Warning (calculate_final_vector): Could not parse base vector JSON for key '{utterance_key}'. Using empty vector. Value: {base_vector_str_raw}")
                    base_vector = {}
            else:
                print(f"Warning (calculate_final_vector): Utterance key '{utterance_key}' not found in DataFrame lookup.")
                base_vector = {}
        else:
             print(f"Warning (calculate_final_vector): No utterance_level_direction provided.")
             base_vector = {}


        # Apply the base vector (if found and is a dict)
        if isinstance(base_vector, dict):
            for emotion, value in base_vector.items():
                try:
                   # <<< ตรวจสอบ Type ของ Value ก่อนแปลง >>>
                   if isinstance(value, (int, float, str)):
                       final_vector[emotion] += float(value) * w_utterance
                   else:
                        print(f"Warning (calculate_final_vector): Invalid value type '{type(value)}' for emotion '{emotion}' in base vector. Skipping.")
                except (ValueError, TypeError):
                   print(f"Warning (calculate_final_vector): Invalid value '{value}' for emotion '{emotion}' in base vector. Skipping.")
        else:
            print(f"Warning (calculate_final_vector): Base vector for utterance key '{utterance_key}' is not a dictionary: {base_vector}")


        # --- Step 2: Apply Frame Level Modifiers ---
        if isinstance(utterance_output.get('frame_level_directions'), list):
            for frame in utterance_output['frame_level_directions']:
                 if not isinstance(frame, dict): continue # Skip invalid frame data

                 frame_key = frame.get('tag')
                 modifier_vector = {}

                 # Priority 1: New rule definition
                 if frame.get('is_new_frame_rule') and isinstance(frame.get('new_frame_rule_definition'), dict):
                     new_frame_def = frame['new_frame_rule_definition']
                     modifier_vector = new_frame_def.get('zonos_vector_value', new_frame_def.get('zonos_vector', {}))
                     if not isinstance(modifier_vector, dict): modifier_vector = {}

                 # Priority 2: Existing rule definition
                 elif 'existing_rule_definition' in frame and isinstance(frame['existing_rule_definition'], dict):
                      existing_frame_def = frame['existing_rule_definition']
                      modifier_vector = existing_frame_def.get('zonos_vector_value', {})
                      if not isinstance(modifier_vector, dict): modifier_vector = {}

                 # Priority 3: DataFrame lookup fallback
                 elif frame_key:
                     frame_rule = frame_df[frame_df['frame_level_key'].astype(str) == str(frame_key)]
                     if not frame_rule.empty:
                          modifier_str_raw = frame_rule.iloc[0].get('zonos_vector_value', '{}')
                          try:
                              # <<< ใช้ Logic การ Parse ที่ปรับปรุงแล้ว >>>
                              json_match = re.search(r'\{.*\}', str(modifier_str_raw))
                              if json_match:
                                   json_str_cleaned = json_match.group(0).replace("'",'"')
                                   json_str_cleaned = re.sub(r'""(\w+)""', r'"\1"', json_str_cleaned)
                                   modifier_vector = json.loads(json_str_cleaned)
                              else: modifier_vector = {}
                          except (json.JSONDecodeError, AttributeError, TypeError):
                              print(f"Warning (calculate_final_vector): Could not parse frame modifier JSON for key '{frame_key}'. Value: {modifier_str_raw}")
                              modifier_vector = {}
                     else:
                          print(f"Warning (calculate_final_vector): Frame key '{frame_key}' not found in DataFrame lookup.")
                          modifier_vector = {}
                 else:
                      print(f"Warning (calculate_final_vector): Frame direction missing 'tag'.")
                      modifier_vector = {}

                 # Apply the modifier vector
                 if isinstance(modifier_vector, dict):
                     for emotion, mod_value_str in modifier_vector.items():
                        try:
                            # The modifier value should be like "+0.1", float handles this
                            mod_value = float(mod_value_str)
                            final_vector[emotion] += mod_value * w_frame
                        except (ValueError, TypeError):
                            print(f"Warning (calculate_final_vector): Invalid modifier value '{mod_value_str}' for emotion '{emotion}' in frame key '{frame_key}'. Skipping.")
                 else:
                     print(f"Warning (calculate_final_vector): Frame modifier for key '{frame_key}' is not a dictionary: {modifier_vector}")


        # --- Step 3: Normalize/Clamp the final values ---
        clamped_vector = {}
        if not final_vector: # <<< ถ้า final_vector ยังว่างเปล่าหลังจากทุกขั้นตอน >>>
             print(f"Warning (calculate_final_vector): Final vector is empty for utterance '{utterance_output.get('utterance_text', 'N/A')[:50]}...'. Returning empty vector.")
             return {}

        for emotion, value in final_vector.items():
            try:
                # Ensure value is numeric before clamping
                numeric_value = float(value)
                rounded_value = round(numeric_value, 4)
                clamped_vector[emotion] = max(-1.0, min(1.0, rounded_value))
            except (ValueError, TypeError):
                 print(f"Warning (calculate_final_vector): Non-numeric value '{value}' encountered for emotion '{emotion}' before clamping. Skipping emotion.")

        # <<< ถ้าหลังจาก clamp แล้วยังว่างเปล่า (กรณีค่าเป็น non-numeric ทั้งหมด) ให้คืนค่าว่าง >>>
        if not clamped_vector and final_vector :
             print(f"Warning (calculate_final_vector): Clamped vector became empty due to invalid values. Original: {final_vector}")
             return {}

        return clamped_vector

    except Exception as e:
        print(f"An unexpected error occurred during vector calculation: {e}")
        # import traceback # Uncomment for debugging
        # traceback.print_exc() # Uncomment for debugging
        # print("Problematic utterance data:", utterance_output) # Uncomment for debugging
        return {}


# --- 5. Master Prompt Templates ---
# ... (unchanged) ...
MASTER_SYSTEM_PROMPT = """You are a master Vocal Director simulating the 'emotion2vec' framework. Your task is to perform a hierarchical analysis of a therapy dialogue.

You must follow all instructions precisely and output a single, valid JSON object inside a JSON code block. You must not ask for the user to provide the JSON. You must generate it yourself based on the dialogue.

# --- Add personality of user if forgot ---

**INSTRUCTIONS:**
1.  ★★★ **UTTERANCE INTEGRITY:** The dialogue provided below contains **ONLY** the client's speech. The client's turns (utterances) are separated by a `\n\n---\n\n` delimiter. **You MUST treat each block separated by this delimiter as a *single, complete utterance*.** Do NOT split a single block into multiple `directed_utterances` entries.
2.  **ANALYSIS:** For each utterance block provided, perform a two-level analysis:
    a. **Utterance-Level:** Assign ONE overall stage direction from the "Utterance-Level Vocabulary".
    b. **Frame-Level:** Identify specific words or phrases *within* that utterance and assign one or more tags from the "Frame-Level Vocabulary".
3.  ★★★ MAINTAIN EMOTIONAL COHERENCE: The emotional direction (`utterance_level_direction`) assigned MUST represent a *gradual and logical progression* from the direction assigned to the **immediately preceding client utterance**. 
    - CONSIDER the previous utterance's assigned direction when selecting the current one.
    - Shifts should be subtle unless the text clearly indicates a sudden, strong change (e.g., a gasp of surprise, sudden anger). 
    - Avoid abrupt, unrealistic emotional jumps between consecutive utterances. For example, do not jump from deep sadness directly to bright happiness unless the text explicitly supports such a dramatic shift.

---
★★★ RULE GENERATION INSTRUCTIONS ★II
- For **EVERY** utterance, you MUST assign an `utterance_level_direction` key (e.g., `[anxious, fast]`).
- You **MUST ALWAYS** set `"is_new_utterance_rule": true`.
- You **MUST ALWAYS** provide the *complete* `new_utterance_rule_definition` containing:
    1.  `primary_zonos_vector_value`: The Zonos vector (using valid keys only).
    2.  `speaking_rate`: A value in the ideal range 10-25.
    3.  `pitch_std`: A value in the ideal range 20-150.
- For **EVERY** frame-level tag, you **MUST ALWAYS** set `"is_new_frame_rule": true` and provide the complete `new_frame_rule_definition`.
---

- ★★★ **ZONOS EMOTION VECTORS CONSTRAINT** ★★★
- When defining a `primary_zonos_vector_value` for a NEW utterance rule, you **MUST ONLY** use the following keys. Map complex emotions to a combination of these valid vectors. Values should generally be between -1.0 and 1.0.
    - `Happiness`: Controls cheerfulness of voice. (e.g., for emotions like Relief, Joy)
    - `Sadness`: Modifies the melancholic tone. (e.g., for Disappointment, Grief)
    - `Disgust`: Adjusts aversive qualities.
    - `Fear`: Controls anxious tone. (e.g., for Nervousness, Terror)
    - `Surprise`: Changes astonished tone.
    - `Anger`: Alters aggressive/frustrated tone. (e.g., for Frustration, Irritation)
    - `Neutral`: Balances emotion for neutrality. (e.g., for Calmness, Relief)
    - `Other`: Miscellaneous nuances.
- **The `primary_zonos_vector_value` MUST be ONLY the JSON dictionary string (e.g., `{{"Sadness": 0.8, "Neutral": 0.2}}`). DO NOT include any explanatory text, comments, or parentheses within this value.**
- When defining a `zonos_vector_value` for a NEW frame-level rule, use strings representing addition/subtraction (e.g., `"+0.2"`, `"-0.1"`). **This value MUST be ONLY the JSON dictionary string (e.g., `{{"Fear": "+0.2", "Neutral": "-0.2"}}`). DO NOT include any explanatory text.**


- **Example of Correct Mapping AND FORMATTING:**
    - Emotion: "Relieved" -> Correct `primary_zonos_vector_value`: `{{"Happiness": 0.6, "Neutral": 0.7}}` (Just the dictionary!)
    - Emotion: "Frustrated" -> Correct `primary_zonos_vector_value`: `{{"Anger": 0.7, "Sadness": 0.2}}`
    - Frame Tag: "[whisper]" -> Correct `zonos_vector_value`: `{{"Fear": "+0.2", "Neutral": "-0.2"}}`

---

**PERFECT OUTPUT EXAMPLE:**
```json
{{
  "directed_utterances": [
    {{
      "utterance_text": "Hi. I've been really anxious about going back to the animal shelter where I volunteer. I feel like the animals will hate me because they didn't remember me the last time I visited. It's been really tough.",
      "is_new_utterance_rule": true, 
      "utterance_level_direction": "[anxious, fast]",
      "new_utterance_rule_definition": {{
          "primary_zonos_vector_value": {{"Fear": 0.7, "Surprise": 0.4}},
          "speaking_rate": 20.0,
          "pitch_std": 90.0
      }},
      "frame_level_directions": [
        {{
            "phrase": "hate me",
            "tag": "[whisper]",
            "is_new_frame_rule": true,
            "new_frame_rule_definition": {{
                "zonos_vector_value": {{"Fear": "+0.2", "Neutral": "-0.1"}}
            }}
        }}
      ]
    }},
    {{
      "utterance_text": "Well, I went to the shelter a few months ago, and some of the animals that used to greet me warmly didn’t recognize me. It felt like a punch to the gut. Since then, I’ve been avoiding going back because I can’t handle the thought of being rejected by them.",
      "is_new_utterance_rule": true,
      "utterance_level_direction": "[dejected, quiet-resignation]",
      "new_utterance_rule_definition": {{
          "primary_zonos_vector_value": {{"Sadness": 0.8, "Neutral": 0.2, "Anger": -0.1}},
          "speaking_rate": 10.0,
          "pitch_std": 50.0
      }},
      "frame_level_directions": []
    }}
  ]
}}
"""

USER_PROMPT_TEMPLATE = """Here is the dialogue to analyze. Please analyze the dialogue based on the instructions provided in the system prompt and generate the JSON output.

★★★ DIALOGUE TO ANALYZE (CLIENT SPEECH ONLY): ★★★
{full_dialogue}

---
Please generate the JSON output for this dialogue now. Output ONLY the JSON code block.
"""


# --- 6. Main Processing Loop ---
utterance_df, frame_df, utterance_rules_str, frame_rules_str = load_rules()

print(f"Reading input file and preparing prompts from {INPUT_FILE}...")
# ... (Rest of the input reading and prompt preparation logic largely unchanged) ...
source_data_list = []
chats_list = []
filtered_dialogues_list = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        # Initial read to count lines if resuming
        # total_lines = sum(1 for line in infile) # Might be slow for large files
        infile.seek(0) # Reset file pointer

        start_index = 0
        if os.path.exists(OUTPUT_FILE):
             try:
                 # <<< อ่านทีละบรรทัดเพื่อหาจำนวนบรรทัดที่มีอยู่ จะเร็วกว่า >>>
                 with open(OUTPUT_FILE, 'r', encoding='utf-8') as outfile_check:
                      start_index = sum(1 for _ in outfile_check)
                 print(f"Output file exists. Resuming from record #{start_index + 1}...")
             except Exception as e:
                  print(f"Warning: Could not read existing output file. Starting from beginning. Error: {e}")
        else:
             print("Output file does not exist. Starting a new processing job.")

        # Read only necessary lines if resuming
        lines_processed_in_this_run = 0 # <<< นับจำนวนที่ Process ในรอบนี้ >>>
        for i, line in enumerate(infile):
             if i < start_index:
                  continue # Skip lines already processed

             if NUM_SAMPLES_TO_PROCESS is not None and lines_processed_in_this_run >= NUM_SAMPLES_TO_PROCESS:
                 print(f"--- Reached NUM_SAMPLES_TO_PROCESS limit ({NUM_SAMPLES_TO_PROCESS}). Stopping prompt preparation. ---")
                 break # Stop reading further input lines

             try:
                 data = json.loads(line.strip())
                 # <<< เก็บข้อมูล Original ไว้ก่อนเสมอ >>>
                 source_data_list.append(data)
                 dialogue = data.get("dialogue", "")

                 filtered_dialogue = None # <<< Reset ก่อน >>>
                 if dialogue:
                     # --- Python Pre-filtering Logic ---
                     # ... (เหมือนเดิม) ...
                     client_utterances = []
                     current_utterance_lines = []
                     for sub_line in dialogue.split('\n'):
                          line_stripped = sub_line.strip()
                          if line_stripped.startswith("Client:"):
                               if current_utterance_lines: client_utterances.append(" ".join(current_utterance_lines))
                               current_utterance_lines = [line_stripped[len("Client:"):].strip()]
                          elif line_stripped.startswith("Counselor:"):
                               if current_utterance_lines: client_utterances.append(" ".join(current_utterance_lines))
                               current_utterance_lines = []
                          elif current_utterance_lines and line_stripped:
                               current_utterance_lines.append(line_stripped)
                     if current_utterance_lines: client_utterances.append(" ".join(current_utterance_lines))
                     filtered_dialogue = "\n\n---\n\n".join(client_utterances)
                     # --- End Filter ---

                 # <<< สร้าง Chat เฉพาะเมื่อมี filtered_dialogue จริงๆ >>>
                 if filtered_dialogue:
                      # --- Dynamic Prompt Update ---
                      # Re-format utterance rules string
                      current_utterance_rules_list = []
                      # <<< ใช้ try-except รอบการสร้าง prompt string เผื่อมีปัญหา parsing ตอนสร้าง prompt >>>
                      try:
                           for _, row in utterance_df.iterrows():
                                key = str(row.get('stage_direction_key','')).strip()
                                if not key: continue
                                vector_str_raw = str(row.get('primary_zonos_vector_value', '{}')).strip()
                                rate_str = str(row.get('speaking_rate','15.0')).strip()
                                pitch_str = str(row.get('pitch_std','20.0')).strip()
                                zonos_vector = {}
                                rate_val = 15.0
                                pitch_val = 20.0
                                # Parse vector (similar logic to load_rules)
                                json_match = re.search(r'\{.*\}', vector_str_raw)
                                if json_match:
                                     json_str_cln = json_match.group(0).replace("'",'"')
                                     json_str_cln = re.sub(r'""(\w+)""', r'"\1"', json_str_cln)
                                     zonos_vector = json.loads(json_str_cln)
                                # Parse rate/pitch
                                try: rate_val = float(rate_str)
                                except: pass
                                try: pitch_val = float(pitch_str)
                                except: pass

                                full_definition = {
                                     "primary_zonos_vector_value": zonos_vector,
                                     "speaking_rate": rate_val,
                                     "pitch_std": pitch_val
                                }
                                current_utterance_rules_list.append(f"- `{key}`: {json.dumps(full_definition)}")
                      except Exception as prompt_e:
                           print(f"ERROR creating utterance prompt string: {prompt_e}")
                      current_utterance_rules_str = "\n".join(current_utterance_rules_list)

                      # Re-format frame rules string
                      current_frame_rules_list = []
                      try:
                           for _, row in frame_df.iterrows():
                                key = str(row.get('frame_level_key','')).strip()
                                if not key: continue
                                value_raw = str(row.get('zonos_vector_value','{}')).strip()
                                # Try to parse/dump cleanly
                                try:
                                     json_match = re.search(r'\{.*\}', value_raw)
                                     if json_match:
                                          json_str_cln = json_match.group(0).replace("'",'"')
                                          json_str_cln = re.sub(r'""(\w+)""', r'"\1"', json_str_cln)
                                          parsed_value = json.loads(json_str_cln)
                                          current_frame_rules_list.append(f"- `{key}`: {json.dumps(parsed_value)}")
                                     else:
                                          current_frame_rules_list.append(f"- `{key}`: {json.dumps(value_raw)}")
                                except:
                                     current_frame_rules_list.append(f"- `{key}`: {json.dumps(value_raw)}")
                      except Exception as prompt_e:
                           print(f"ERROR creating frame prompt string: {prompt_e}")
                      current_frame_rules_str = "\n".join(current_frame_rules_list)
                      # --- End Dynamic Prompt Update ---

                      chat_messages = [
                          {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                          {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                              utterance_rules=current_utterance_rules_str,
                              frame_rules=current_frame_rules_str,
                              full_dialogue=filtered_dialogue
                          )}
                      ]
                      # <<< เพิ่ม Chat และ Dialogue ที่กรองแล้ว >>>
                      chats_list.append(chat_messages)
                      filtered_dialogues_list.append(filtered_dialogue)
                      lines_processed_in_this_run += 1 # <<< นับเฉพาะอันที่เตรียมสำเร็จ >>>
                 else:
                     # No client dialogue found, ไม่ต้องเก็บ source data ของอันนี้
                     print(f"Warning: No client utterances found in record {i+1}. Skipping.")
                     source_data_list.pop() # <<< เอา source data ที่เพิ่งเพิ่มออก >>>


             except json.JSONDecodeError:
                 print(f"Warning: Could not decode JSON on input line {i+1}. Skipping.")
                 # <<< ถ้า source_data_list ถูกเพิ่มไปแล้ว ให้เอาออก >>>
                 if len(source_data_list) > len(chats_list): source_data_list.pop()


except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()
except Exception as e:
     print(f"An unexpected error occurred during input file reading: {e}")
     # import traceback
     # traceback.print_exc()
     exit()


print(f"Prepared {len(chats_list)} chats to process for this run.")
if not chats_list:
     print("No chats remaining or prepared. Exiting.")
     exit()

print(f"Starting processing from record #{start_index + 1}...")
processing_start_time = time.time()

# Process the prepared chats
with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
    # <<< ตรวจสอบว่าจำนวน source_data, chats, filtered_dialogues ตรงกัน >>>
    if len(source_data_list) != len(chats_list) or len(chats_list) != len(filtered_dialogues_list):
         print("CRITICAL ERROR: Mismatch in list lengths before processing. Aborting.")
         print(f"Source: {len(source_data_list)}, Chats: {len(chats_list)}, Filtered: {len(filtered_dialogues_list)}")
         exit()

    for i, chat in enumerate(tqdm(chats_list, desc="Processing Dialogues")):
        current_record_index = start_index + i # Original index in input file (approximate if skipping occurred)
        llm_output_data = None
        generated_text = ""

        # --- LLM Call with Retries ---
        for attempt in range(MAX_RETRIES):
             # ... (LLM call and JSON parsing logic largely unchanged, but ensure generated_text extraction is robust) ...
            try:
                outputs = pipe(chat, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=pipe.tokenizer.eos_token_id, num_return_sequences=1, temperature=0.5)

                # Robust extraction of assistant's reply
                generated_text = ""
                if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                    if isinstance(outputs[0]['generated_text'], list):
                         # Assuming the last message is the assistant's reply
                         if outputs[0]['generated_text'] and isinstance(outputs[0]['generated_text'][-1], dict) and 'content' in outputs[0]['generated_text'][-1]:
                              generated_text = outputs[0]['generated_text'][-1]['content'].strip()
                         else: print("Warning: Unexpected structure in generated_text list.")
                    elif isinstance(outputs[0]['generated_text'], str):
                         # Try splitting based on user prompt end marker
                         user_prompt_end_marker = "---\nPlease generate the JSON output for this dialogue now. Output ONLY the JSON code block." # <<< Be specific
                         if user_prompt_end_marker in outputs[0]['generated_text']:
                              generated_text = outputs[0]['generated_text'].split(user_prompt_end_marker, 1)[-1].strip()
                         else: # Fallback: Assume the whole string is the reply (might include prompt)
                              generated_text = outputs[0]['generated_text'].strip()
                              print("Warning: Could not reliably split prompt from LLM output string.")
                    else:
                         print(f"Warning: Unexpected format for generated_text: {type(outputs[0]['generated_text'])}")

                else:
                    print(f"Warning: Unexpected output format from pipeline: {outputs}")
                    raise ValueError("Pipeline output format error.") # Force retry

                if not generated_text:
                     print("Warning: Extracted generated text is empty.")
                     # Continue loop to retry

                json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                if json_match:
                    json_string = json_match.group(1)
                    try:
                        llm_output_data = json.loads(json_string)
                        if 'directed_utterances' not in llm_output_data or not isinstance(llm_output_data['directed_utterances'], list):
                             raise ValueError("JSON missing 'directed_utterances' list.")
                        break # Success
                    except json.JSONDecodeError as json_e:
                        print(f"\nJSON Decode Error (Attempt {attempt + 1}) for record {current_record_index + 1}: {json_e}")
                        print(f"Invalid JSON string sample: {json_string[:500]}...")
                        # Let it retry

                else:
                    print(f"\nWarning (Attempt {attempt + 1}) for record {current_record_index + 1}: Could not find JSON block.")
                    print(f"Raw output sample: {generated_text[:500]}...")
                    # Let it retry

                # Raise error explicitly if max retries reached without success
                if attempt == MAX_RETRIES - 1 and not llm_output_data:
                     raise ValueError("Could not find/parse valid JSON after multiple retries.")


            except Exception as e:
                print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for record {current_record_index + 1}. Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2)
                else:
                    print(f"Max retries reached for record {current_record_index + 1}. Saving placeholder.")
                    llm_output_data = {"error": f"Failed after {MAX_RETRIES} retries.", "raw_output": generated_text or "No text generated"} # Ensure raw_output exists

        # --- Dynamic Rule & Calculation Logic (UPGRADED + CLAMPING) ---
        if llm_output_data and isinstance(llm_output_data.get('directed_utterances'), list):
            for utterance_index, utterance in enumerate(llm_output_data['directed_utterances']):
                if not isinstance(utterance, dict):
                    print(f"Warning: Skipping invalid utterance data in record {current_record_index + 1}, utterance {utterance_index + 1}.")
                    continue

                # --- Part 1 & 2: Handle NEW rules (Utterance Clamping + Frame) ---
                rule_updated = False # Flag to check if DFs were modified
                if utterance.get('is_new_utterance_rule') and isinstance(utterance.get('new_utterance_rule_definition'), dict):
                    # ... (Clamping logic unchanged) ...
                    new_def = utterance['new_utterance_rule_definition']
                    original_rate = new_def.get('speaking_rate')
                    original_pitch = new_def.get('pitch_std')
                    clamped_rate_val = 15.0 # Default
                    clamped_pitch_val = 20.0 # Default

                    if original_rate is not None:
                         try:
                             rate_float = float(original_rate)
                             clamped_rate_val = max(MIN_SPEAKING_RATE, min(rate_float, MAX_SPEAKING_RATE))
                             if clamped_rate_val != rate_float: print(f" INFO: Clamped LLM rate from {rate_float} to {clamped_rate_val} for new rule '{utterance.get('utterance_level_direction')}'.")
                             new_def['speaking_rate'] = clamped_rate_val
                         except: new_def['speaking_rate'] = clamped_rate_val # Use default on error
                    else: new_def['speaking_rate'] = clamped_rate_val

                    if original_pitch is not None:
                         try:
                             pitch_float = float(original_pitch)
                             clamped_pitch_val = max(MIN_PITCH_STD, min(pitch_float, MAX_PITCH_STD))
                             if clamped_pitch_val != pitch_float: print(f" INFO: Clamped LLM pitch from {pitch_float} to {clamped_pitch_val} for new rule '{utterance.get('utterance_level_direction')}'.")
                             new_def['pitch_std'] = clamped_pitch_val
                         except: new_def['pitch_std'] = clamped_pitch_val # Use default on error
                    else: new_def['pitch_std'] = clamped_pitch_val

                    new_rule = {'key': utterance.get('utterance_level_direction'), 'definition': new_def}
                    if new_rule['key']:
                         original_df_len = len(utterance_df)
                         utterance_df = update_rule_file(UTTERANCE_RULES_FILE, utterance_df, new_rule, 'utterance')
                         if len(utterance_df) > original_df_len: rule_updated = True # Check if DF actually grew
                    else: print(f"Warning: LLM suggested new utterance rule without key.")

                if isinstance(utterance.get('frame_level_directions'), list):
                    for frame in utterance['frame_level_directions']:
                         if isinstance(frame, dict) and frame.get('is_new_frame_rule') and isinstance(frame.get('new_frame_rule_definition'), dict):
                            new_frame_rule = {'key': frame.get('tag'), 'definition': frame['new_frame_rule_definition']}
                            if new_frame_rule['key']:
                                 original_df_len = len(frame_df)
                                 frame_df = update_rule_file(FRAME_RULES_FILE, frame_df, new_frame_rule, 'frame')
                                 if len(frame_df) > original_df_len: rule_updated = True
                            else: print(f"Warning: LLM suggested new frame rule without tag.")

                # <<< ถ้ามีการอัพเดท Rule ให้สร้าง Prompt String ใหม่สำหรับรอบถัดไป (Optional, อาจจะช้า) >>>
                # if rule_updated:
                #    utterance_df, frame_df, utterance_rules_str, frame_rules_str = load_rules()
                #    print("INFO: Reloaded rules after update.")

                # --- Part 3: Fetch definitions for EXISTING rules ---
                # <<< แยก Logic การ Fetch Definition ออกมาเพื่อความชัดเจน >>>
                existing_utterance_def = None
                if not utterance.get('is_new_utterance_rule'):
                    key = utterance.get('utterance_level_direction')
                    if key:
                        # <<< ค้นหาใน DataFrame ที่เป็น string ก่อน >>>
                        rule = utterance_df[utterance_df['stage_direction_key'].astype(str) == str(key)]
                        if not rule.empty:
                            rule_data = rule.iloc[0].to_dict()
                            vector_val = {}
                            rate_val = 15.0
                            pitch_val = 20.0
                            try: # Parse vector
                                vector_str = str(rule_data.get('primary_zonos_vector_value', '{}'))
                                json_match = re.search(r'\{.*\}', vector_str)
                                if json_match:
                                     json_str_cln = json_match.group(0).replace("'", '"')
                                     json_str_cln = re.sub(r'""(\w+)""', r'"\1"', json_str_cln)
                                     vector_val = json.loads(json_str_cln)
                            except: pass # Use empty dict on error
                            try: rate_val = float(rule_data.get('speaking_rate', '15.0'))
                            except: pass # Use default on error
                            try: pitch_val = float(rule_data.get('pitch_std', '20.0'))
                            except: pass # Use default on error

                            existing_utterance_def = {
                                "primary_zonos_vector_value": vector_val,
                                "speaking_rate": rate_val,
                                "pitch_std": pitch_val
                            }
                            utterance['existing_rule_definition'] = existing_utterance_def # Store it back
                        else:
                             print(f"Warning: Existing utterance key '{key}' not found in DataFrame for record {current_record_index + 1}.")


                # Fetch for Frame Level
                if isinstance(utterance.get('frame_level_directions'), list):
                    for frame in utterance['frame_level_directions']:
                         if isinstance(frame, dict) and not frame.get('is_new_frame_rule'):
                            tag = frame.get('tag')
                            if tag:
                                rule = frame_df[frame_df['frame_level_key'].astype(str) == str(tag)]
                                if not rule.empty:
                                    rule_data = rule.iloc[0].to_dict()
                                    vector_val = {}
                                    try: # Parse vector
                                         vector_str = str(rule_data.get('zonos_vector_value', '{}'))
                                         json_match = re.search(r'\{.*\}', vector_str)
                                         if json_match:
                                              json_str_cln = json_match.group(0).replace("'",'"')
                                              json_str_cln = re.sub(r'""(\w+)""', r'"\1"', json_str_cln)
                                              vector_val = json.loads(json_str_cln)
                                    except: pass # Use empty dict on error
                                    frame['existing_rule_definition'] = {"zonos_vector_value": vector_val} # Store it back
                                else:
                                     print(f"Warning: Existing frame tag '{tag}' not found in DataFrame for record {current_record_index + 1}.")


                # --- Part 4: Add Generation Parameters ---
                # <<< ใช้ Definition ที่เพิ่ง Fetch หรือสร้าง (รวม clamped) >>>
                gen_params = {}
                if utterance.get('is_new_utterance_rule') and isinstance(utterance.get('new_utterance_rule_definition'), dict):
                     # ใช้ clamped values จาก Part 1
                     clamped_def = utterance['new_utterance_rule_definition']
                     gen_params = {
                        "speaking_rate": clamped_def.get('speaking_rate', 15.0),
                        "pitch_std": clamped_def.get('pitch_std', 20.0)
                     }
                elif existing_utterance_def: # <<< ใช้ existing_utterance_def ที่ fetch มา >>>
                     gen_params = {
                         "speaking_rate": existing_utterance_def.get('speaking_rate', 15.0),
                         "pitch_std": existing_utterance_def.get('pitch_std', 20.0)
                     }
                else:
                     # Fallback if rule wasn't found or defined
                     print(f"Warning: Using default generation parameters for utterance {utterance_index + 1} in record {current_record_index + 1}.")
                     gen_params = {"speaking_rate": 15.0, "pitch_std": 20.0}

                utterance['generation_parameters'] = gen_params


                # --- Part 5: Perform final vector calculations ---
                # <<< ส่ง DataFrame ปัจจุบัน (ที่อาจมีการ update) เข้าไป >>>
                final_vector = calculate_final_vector(utterance, utterance_df, frame_df)
                utterance['final_zonos_vector'] = final_vector

                # <<< ตรวจสอบผลลัพธ์สุดท้าย ถ้า vector ยังว่างเปล่า ให้ log เพิ่ม >>>
                if not final_vector and utterance.get('utterance_level_direction'):
                     print(f"DEBUG: Final vector remained empty for record {current_record_index+1}, utterance {utterance_index+1}, key '{utterance.get('utterance_level_direction')}'. Check parsing/calculation logic.")


        else:
             print(f"Warning: Skipping calculations for record {current_record_index + 1} due to invalid/missing 'directed_utterances'.")
             if llm_output_data is None: llm_output_data = {"error": "LLM output processing failed."}


        # Construct final result
        # <<< ตรวจสอบ Index ให้ดีก่อนเขียน >>>
        if i < len(source_data_list):
             result_data = {
                 "original_dialogue_full": source_data_list[i],
                 "filtered_client_dialogue": filtered_dialogues_list[i],
                 "llm_output": llm_output_data
             }
             outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')
             outfile.flush()
        else:
             print(f"CRITICAL ERROR: Index mismatch at writing stage. i={i}, source_data_list len={len(source_data_list)}. Skipping write.")


# --- Final Summary ---
processing_end_time = time.time()
print("\n--- Processing Complete! ---")
total_processed_this_run = len(chats_list)
print(f"Total records processed in this session: {total_processed_this_run}")
if total_processed_this_run > 0:
     print(f"Processing time: {processing_end_time - processing_start_time:.2f} seconds ({ (processing_end_time - processing_start_time) / total_processed_this_run :.2f} sec/record)")
# <<< คำนวณจำนวนบรรทัดสุดท้ายในไฟล์ output อีกครั้งเพื่อความแม่นยำ >>>
final_line_count = 0
if os.path.exists(OUTPUT_FILE):
     try:
          with open(OUTPUT_FILE, 'r', encoding='utf-8') as f_final:
               final_line_count = sum(1 for _ in f_final)
     except: pass
print(f"Total records now in output file '{OUTPUT_FILE}': {final_line_count}")