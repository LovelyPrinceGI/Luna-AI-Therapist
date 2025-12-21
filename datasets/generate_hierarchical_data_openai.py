# --- 1. Imports ---
import json
import time
from tqdm import tqdm
import pandas as pd
import re
from collections import defaultdict
import os 
from openai import OpenAI # <<< NEW: Import OpenAI

# --- 2. Configuration ---
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # <<< REMOVED
OPENAI_MODEL_ID = "gpt-4o-mini" # <<< NEW: Use GPT-4o Mini
UTTERANCE_RULES_FILE = "utterance_level_rules.csv"
FRAME_RULES_FILE = "frame_level_rules.csv"
INPUT_FILE = "cactus_huggingface_train.jsonl"
OUTPUT_FILE = "final_hierarchical_output_v_openai.jsonl" # <<< NEW: Changed output filename
# MAX_NEW_TOKENS = 8192 # <<< REMOVED (Not needed for OpenAI API in the same way)
NUM_SAMPLES_TO_PROCESS = 20
MAX_RETRIES = 5 # Reduced retries, API is more stable

# Zonos clamping parameters (unchanged)
MIN_SPEAKING_RATE = 10.0
MAX_SPEAKING_RATE = 25.0
MIN_PITCH_STD = 20.0
MAX_PITCH_STD = 150.0

# --- 3. Setup OpenAI Client ---
print("Setting up OpenAI client...")
if "OPENAI_API_KEY" not in os.environ:
    print("="*50)
    print("Error: OPENAI_API_KEY not found in environment variables.")
    print("Please set the environment variable before running:")
    print("export OPENAI_API_KEY='your_api_key_here'")
    print("="*50)
    exit()

# This automatically reads the API key from the environment variable
client = OpenAI()
print(f"OpenAI client initialized for model: {OPENAI_MODEL_ID}")


# --- 4. Load and Format Rules from CSV Files ---
# <<< This function is kept to load DataFrames for update_rule_file and calculate_final_vector >>>
def load_rules():
    """Loads rules from CSVs into DataFrames. (Prompt string generation is no longer used)"""
    print("Loading rules from CSV files (for DataFrame reference)...")
    try:
        if not os.path.exists(UTTERANCE_RULES_FILE):
             # Create file with header if it doesn't exist
             pd.DataFrame(columns=['stage_direction_key', 'primary_zonos_vector_value', 'speaking_rate', 'pitch_std']).to_csv(UTTERANCE_RULES_FILE, index=False)
             print(f"Created empty {UTTERANCE_RULES_FILE}")
        if not os.path.exists(FRAME_RULES_FILE):
             pd.DataFrame(columns=['frame_level_key', 'zonos_vector_value']).to_csv(FRAME_RULES_FILE, index=False)
             print(f"Created empty {FRAME_RULES_FILE}")

        utterance_df = pd.read_csv(UTTERANCE_RULES_FILE, dtype=str).fillna('')
        frame_df = pd.read_csv(FRAME_RULES_FILE, dtype=str).fillna('')

        print("Rules DataFrames loaded successfully.")
        # We no longer need to return the strings for the prompt
        return utterance_df, frame_df, "", "" # Return empty strings

    except Exception as e:
        print(f"An unexpected error occurred during rule loading: {e}")
        exit()

# --- 5. Automatic Rule Updater ---
# ... (update_rule_file function unchanged, it's still needed to log new rules) ...
def update_rule_file(file_path, df, new_rule_data, rule_type):
    """
    Appends a new rule to the specified CSV file and updates the in-memory DataFrame.
    (Assumes values in new_rule_data['definition'] are already clamped/validated)
    """
    try:
        # print(f"\n✨ New {'utterance' if rule_type == 'utterance' else 'frame'} rule! Updating {file_path}...") # <<< Optional: make less verbose

        key = new_rule_data.get('key')
        definition = new_rule_data.get('definition', {})
        if not key:
             print("❌ Error: Cannot update rule file, key is missing.")
             return df

        key_column = 'stage_direction_key' if rule_type == 'utterance' else 'frame_level_key'
        
        # Check if the key already exists
        if key in df[key_column].astype(str).values:
             # print(f"ℹ️ Rule '{key}' already exists in DataFrame. Skipping concat.") # <<< Optional: make less verbose
             return df # Just return the original df
        
        # If new, prepare and append
        if rule_type == 'utterance':
            zonos_vector_csv_str = json.dumps(definition.get('primary_zonos_vector_value', definition.get('zonos_vector', {})))
            new_row_data = {
                'stage_direction_key': key,
                'primary_zonos_vector_value': zonos_vector_csv_str,
                'speaking_rate': definition.get('speaking_rate', 15.0),
                'pitch_std': definition.get('pitch_std', 20.0)
            }
        elif rule_type == 'frame':
            zonos_vector_value_csv_str = json.dumps(definition.get('zonos_vector_value', definition.get('zonos_vector', {})))
            new_row_data = {
                'frame_level_key': key,
                'zonos_vector_value': zonos_vector_value_csv_str
            }
        
        new_row = pd.DataFrame([new_row_data])
        # Append to CSV (header=False because we ensured it exists in load_rules)
        new_row.to_csv(file_path, mode='a', header=False, index=False)

        # Append to in-memory DataFrame
        new_row_aligned = pd.DataFrame([new_row_data], columns=df.columns)
        updated_df = pd.concat([df, new_row_aligned], ignore_index=True)
        print(f"✅ Successfully added new rule to CSV and DataFrame: {key}") # <<< Keep this log
        
        return updated_df

    except Exception as e:
        print(f"❌ Error updating rule file {file_path}: {e}")
        return df

# --- 6. Vector Calculation Function ---
# ... (calculate_final_vector function unchanged, it's still needed) ...
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
        # <<< MODIFIED: We ONLY trust the new_utterance_rule_definition, as per our prompt >>>
        if utterance_output.get('is_new_utterance_rule') and isinstance(utterance_output.get('new_utterance_rule_definition'), dict):
            new_def = utterance_output['new_utterance_rule_definition']
            base_vector = new_def.get('primary_zonos_vector_value', new_def.get('zonos_vector', {}))
            if not isinstance(base_vector, dict): base_vector = {} # Ensure it's a dict
        else:
            # This case *shouldn't* happen if the new prompt is followed
            print(f"Warning (calculate_final_vector): 'is_new_utterance_rule' was false or definition missing for '{utterance_output.get('utterance_level_direction')}'. Vector will be empty.")
            base_vector = {}
        # <<< REMOVED: Fallback logic to search DataFrame (Priority 2 & 3) >>>

        # Apply the base vector (if found and is a dict)
        if isinstance(base_vector, dict):
            for emotion, value in base_vector.items():
                try:
                   if isinstance(value, (int, float, str)):
                       final_vector[emotion] += float(value) * w_utterance
                   else:
                        print(f"Warning (calculate_final_vector): Invalid value type '{type(value)}' for emotion '{emotion}' in base vector. Skipping.")
                except (ValueError, TypeError):
                   print(f"Warning (calculate_final_vector): Invalid value '{value}' for emotion '{emotion}' in base vector. Skipping.")
        
        # --- Step 2: Apply Frame Level Modifiers ---
        if isinstance(utterance_output.get('frame_level_directions'), list):
            for frame in utterance_output['frame_level_directions']:
                 if not isinstance(frame, dict): continue

                 modifier_vector = {}
                 # <<< MODIFIED: We ONLY trust the new_frame_rule_definition >>>
                 if frame.get('is_new_frame_rule') and isinstance(frame.get('new_frame_rule_definition'), dict):
                     new_frame_def = frame['new_frame_rule_definition']
                     modifier_vector = new_frame_def.get('zonos_vector_value', new_frame_def.get('zonos_vector', {}))
                     if not isinstance(modifier_vector, dict): modifier_vector = {}
                 else:
                     print(f"Warning (calculate_final_vector): 'is_new_frame_rule' was false or definition missing for tag '{frame.get('tag')}'.")
                     modifier_vector = {}
                 # <<< REMOVED: Fallback logic to search DataFrame (Priority 2 & 3) >>>

                 # Apply the modifier vector
                 if isinstance(modifier_vector, dict):
                     for emotion, mod_value_str in modifier_vector.items():
                        try:
                            mod_value = float(mod_value_str)
                            final_vector[emotion] += mod_value * w_frame
                        except (ValueError, TypeError):
                            print(f"Warning (calculate_final_vector): Invalid modifier value '{mod_value_str}' for emotion '{emotion}' in frame key '{frame.get('tag')}'. Skipping.")

        # --- Step 3: Normalize/Clamp the final values ---
        clamped_vector = {}
        if not final_vector:
             # This is okay if the base vector was truly empty (e.g., [neutral])
             # print(f"Note (calculate_final_vector): Final vector is empty for utterance '{utterance_output.get('utterance_text', 'N/A')[:50]}...'.")
             return {}

        for emotion, value in final_vector.items():
            try:
                numeric_value = float(value)
                rounded_value = round(numeric_value, 4)
                clamped_vector[emotion] = max(-1.0, min(1.0, rounded_value))
            except (ValueError, TypeError):
                 print(f"Warning (calculate_final_vector): Non-numeric value '{value}' for emotion '{emotion}' before clamping. Skipping.")

        return clamped_vector

    except Exception as e:
        print(f"An unexpected error occurred during vector calculation: {e}")
        return {}


# --- 7. Master Prompt Templates ---
# <<< Using the EXACT prompt we perfected >>>
MASTER_SYSTEM_PROMPT = """You are a master Vocal Director simulating the 'emotion2vec' framework. Your task is to perform a hierarchical analysis of a therapy dialogue.

You must follow all instructions precisely and output a single, valid JSON object inside a JSON code block. You must not ask for the user to provide the JSON. You must generate it yourself based on the dialogue.

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

- ★★★ **ZONOS EMOTION VECTORS CONSTRAINT** ★II
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


# --- 8. Main Processing Loop ---
# <<< We still load the DFs to pass them to update/calculate functions >>>
utterance_df, frame_df, _, _ = load_rules() 

# --- CONFIGURATION FOR SIMULATION ---
MAX_TURNS = 10  # จำนวนรอบที่จะคุยกัน
SIMULATION_OUTPUT_FILE = "simulation_session_sarah_v1.jsonl"
CLIENT_PERSONA_PROMPT = """You are acting as a role-play client in a Cognitive Behavioral Therapy (CBT) session.

**YOUR PERSONA:**
- Name: Sarah
- Condition: Moderate Depression and Social Anxiety.
- Current State: You feel hopeless about your job search and assume everyone judges you.
- Interaction Style: You are initially resistant and brief. You only open up if the therapist shows genuine empathy.

**TASK:**
1. Analyze the Therapist's latest input.
2. Determine how this affects your emotional state.
3. GENERATE your response (as the client).
4. ASSIGN the vocal style (Zonos Vectors) for your response based on your NEW emotional state.

**OUTPUT FORMAT:**
You must output a JSON object strictly following this structure:
{
  "thought_process": "Brief reasoning...",
  "client_response_text": "Text response...",
  "directed_utterance": {
      "utterance_level_direction": "[emotion, speed]",
      "is_new_utterance_rule": true,
      "new_utterance_rule_definition": {
          "primary_zonos_vector_value": {"Sadness": 0.8, "Neutral": 0.2},
          "speaking_rate": 15.0,
          "pitch_std": 20.0
      },
      "frame_level_directions": []
  }
}
"""

print(f"Starting Simulation Mode... (Max Turns: {MAX_TURNS})")

# 1. Initialize History & State
history = [] 
turn_count = 0

# 2. Start the Loop
with open(SIMULATION_OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
    
    while turn_count < MAX_TURNS:
        print(f"\n--- Turn {turn_count + 1} ---")
        
        # --- STEP A: THERAPIST TURN (Mirror Model Placeholder) ---
        # ตรงนี้เดี๋ยวไดจิเอา Model Therapist ของจริงมาใส่
        # ตอนนี้ลูน่าขอ Mockup (จำลอง) เป็นหมอให้ก่อนนะ
        if turn_count == 0:
            therapist_msg = "Hello Sarah, how are you feeling today?"
        else:
            # ในอนาคต: therapist_msg = mirror_model.generate(history)
            therapist_msg = input("Enter Therapist Message (or press Enter for auto): ") or "I hear that you are struggling. Can you tell me more?"
        
        print(f"👩‍⚕️ Therapist: {therapist_msg}")
        history.append({"role": "counselor", "content": therapist_msg})

        # --- STEP B: CLIENT TURN (OpenAI Simulator) ---
        # เตรียม Prompt สำหรับ Client Agent
        client_messages = [
            {"role": "system", "content": CLIENT_PERSONA_PROMPT},
            {"role": "user", "content": f"Therapist just said: '{therapist_msg}'. \n\nConversation History: {json.dumps(history[-4:])}"}
        ]

        llm_output_data = None
        
        # Call GPT-4o-mini (Client Agent)
        try:
            completion = client.chat.completions.create(
                model=OPENAI_MODEL_ID,
                messages=client_messages,
                temperature=0.7, # ให้มีความ creative หน่อยในการ roleplay
                response_format={"type": "json_object"}
            )
            generated_text = completion.choices[0].message.content
            llm_output_data = json.loads(generated_text)

        except Exception as e:
            print(f"Error calling OpenAI for Client Simulator: {e}")
            break

        # --- STEP C: PROCESS ZONOS VECTORS (Logic เดิมของไดจิ) ---
        if llm_output_data and "directed_utterance" in llm_output_data:
            utterance = llm_output_data["directed_utterance"]
            
            # 1. Update Rules (Utterance Level)
            if utterance.get('is_new_utterance_rule'):
                new_rule = {'key': utterance.get('utterance_level_direction'), 'definition': utterance['new_utterance_rule_definition']}
                utterance_df = update_rule_file(UTTERANCE_RULES_FILE, utterance_df, new_rule, 'utterance')

            # 2. Update Rules (Frame Level)
            if isinstance(utterance.get('frame_level_directions'), list):
                for frame in utterance['frame_level_directions']:
                     if frame.get('is_new_frame_rule'):
                        new_frame_rule = {'key': frame.get('tag'), 'definition': frame['new_frame_rule_definition']}
                        frame_df = update_rule_file(FRAME_RULES_FILE, frame_df, new_frame_rule, 'frame')
            
            # 3. Calculate Final Vector
            final_vector = calculate_final_vector(utterance, utterance_df, frame_df)
            utterance['final_zonos_vector'] = final_vector
            
            # Show Result
            client_text = llm_output_data.get("client_response_text", "")
            print(f"🤒 Client ({utterance['utterance_level_direction']}): {client_text}")
            
            # Update History
            history.append({"role": "client", "content": client_text})

            # Save to File
            result_data = {
                "turn_id": turn_count,
                "therapist_input": therapist_msg,
                "client_output": llm_output_data
            }
            outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')
            outfile.flush()

        turn_count += 1
        time.sleep(1) # พักหายใจนิดนึง

print("\n--- Simulation Complete! ---")