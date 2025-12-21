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

# <<< NEW USER_PROMPT_TEMPLATE (no rules) >>>
USER_PROMPT_TEMPLATE = """Here is the dialogue to analyze. Please analyze the dialogue based on the instructions provided in the system prompt and generate the JSON output.

★★★ DIALOGUE TO ANALYZE (CLIENT SPEECH ONLY): ★★★
{full_dialogue}

---
Please generate the JSON output for this dialogue now. Output ONLY the JSON code block.
"""


# --- 8. Main Processing Loop ---
# <<< We still load the DFs to pass them to update/calculate functions >>>
utterance_df, frame_df, _, _ = load_rules() 

print(f"Reading input file and preparing prompts from {INPUT_FILE}...")
# ... (Input reading logic unchanged) ...
source_data_list = []
chats_list = []
filtered_dialogues_list = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        start_index = 0
        if os.path.exists(OUTPUT_FILE):
             try:
                 with open(OUTPUT_FILE, 'r', encoding='utf-8') as outfile_check:
                      start_index = sum(1 for _ in outfile_check)
                 print(f"Output file exists. Resuming from record #{start_index + 1}...")
             except Exception as e:
                  print(f"Warning: Could not read existing output file. Starting from beginning. Error: {e}")
        else:
             print("Output file does not exist. Starting a new processing job.")

        lines_processed_in_this_run = 0
        for i, line in enumerate(infile):
             if i < start_index:
                  continue 

             if NUM_SAMPLES_TO_PROCESS is not None and lines_processed_in_this_run >= NUM_SAMPLES_TO_PROCESS:
                 print(f"--- Reached NUM_SAMPLES_TO_PROCESS limit ({NUM_SAMPLES_TO_PROCESS}). Stopping prompt preparation. ---")
                 break 

             try:
                 data = json.loads(line.strip())
                 source_data_list.append(data)
                 dialogue = data.get("dialogue", "")

                 filtered_dialogue = None 
                 if dialogue:
                     # --- Python Pre-filtering Logic (unchanged) ---
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

                 if filtered_dialogue:
                      # <<< NO MORE Dynamic Prompt Update needed! >>>

                      chat_messages = [
                          {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                          {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                              # No more rules passed in!
                              full_dialogue=filtered_dialogue
                          )}
                      ]
                      chats_list.append(chat_messages)
                      filtered_dialogues_list.append(filtered_dialogue)
                      lines_processed_in_this_run += 1
                 else:
                     print(f"Warning: No client utterances found in record {i+1}. Skipping.")
                     source_data_list.pop()

             except json.JSONDecodeError:
                 print(f"Warning: Could not decode JSON on input line {i+1}. Skipping.")
                 if len(source_data_list) > len(chats_list): source_data_list.pop()

except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()
except Exception as e:
     print(f"An unexpected error occurred during input file reading: {e}")
     exit()


print(f"Prepared {len(chats_list)} chats to process for this run.")
if not chats_list:
     print("No chats remaining or prepared. Exiting.")
     exit()

print(f"Starting processing from record #{start_index + 1}...")
processing_start_time = time.time()

# Process the prepared chats
with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
    if len(source_data_list) != len(chats_list) or len(chats_list) != len(filtered_dialogues_list):
         print("CRITICAL ERROR: Mismatch in list lengths before processing. Aborting.")
         exit()

    for i, chat in enumerate(tqdm(chats_list, desc="Processing Dialogues")):
        current_record_index = start_index + i
        llm_output_data = None
        generated_text = ""

        # --- <<< NEW: LLM Call with Retries (OpenAI) >>> ---
        for attempt in range(MAX_RETRIES):
            try:
                completion = client.chat.completions.create(
                    model=OPENAI_MODEL_ID,
                    messages=chat,
                    temperature=0.0, # Force deterministic JSON output
                    response_format={"type": "json_object"} # Force JSON mode
                )
                
                generated_text = completion.choices[0].message.content
                
                if not generated_text:
                    raise ValueError("OpenAI returned an empty message content.")
                    
                # Try parsing the JSON immediately (it should be valid)
                try:
                    llm_output_data = json.loads(generated_text)
                    if 'directed_utterances' not in llm_output_data or not isinstance(llm_output_data['directed_utterances'], list):
                         raise ValueError("JSON missing 'directed_utterances' list.")
                    break # Success!
                except json.JSONDecodeError as json_e:
                    print(f"\nJSON Decode Error (Attempt {attempt + 1}) for record {current_record_index + 1}: {json_e}")
                    print(f"Invalid JSON string from API: {generated_text[:500]}...")
                    # This is bad if it happens, as JSON mode should prevent it
                    # Let it retry
            
            except Exception as e:
                # Catch OpenAI API errors (RateLimitError, APIError, etc.)
                print(f"\nOpenAI API Error (Attempt {attempt + 1}/{MAX_RETRIES}) for record {current_record_index + 1}. Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5) # Wait 5 seconds before retrying API
                else:
                    print(f"Max retries reached for record {current_record_index + 1}. Saving placeholder.")
                    llm_output_data = {"error": f"Failed after {MAX_RETRIES} retries. Last error: {str(e)}", "raw_output": "API call failed"}
        # --- <<< END OF NEW LLM CALL >>> ---


        # --- Dynamic Rule & Calculation Logic (UPGRADED + CLAMPING) ---
        # <<< This part remains mostly the same, it just consumes llm_output_data >>>
        if llm_output_data and isinstance(llm_output_data.get('directed_utterances'), list):
            for utterance_index, utterance in enumerate(llm_output_data['directed_utterances']):
                if not isinstance(utterance, dict):
                    print(f"Warning: Skipping invalid utterance data in record {current_record_index + 1}, utterance {utterance_index + 1}.")
                    continue

                # Part 1 & 2: Handle NEW rules (Clamping + Updating CSV)
                # <<< This logic now runs for EVERY utterance, as per our prompt >>>
                if utterance.get('is_new_utterance_rule') and isinstance(utterance.get('new_utterance_rule_definition'), dict):
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
                         except: new_def['speaking_rate'] = clamped_rate_val
                    else: new_def['speaking_rate'] = clamped_rate_val

                    if original_pitch is not None:
                         try:
                             pitch_float = float(original_pitch)
                             clamped_pitch_val = max(MIN_PITCH_STD, min(pitch_float, MAX_PITCH_STD))
                             if clamped_pitch_val != pitch_float: print(f" INFO: Clamped LLM pitch from {pitch_float} to {clamped_pitch_val} for new rule '{utterance.get('utterance_level_direction')}'.")
                             new_def['pitch_std'] = clamped_pitch_val
                         except: new_def['pitch_std'] = clamped_pitch_val
                    else: new_def['pitch_std'] = clamped_pitch_val

                    new_rule = {'key': utterance.get('utterance_level_direction'), 'definition': new_def}
                    if new_rule['key']:
                         utterance_df = update_rule_file(UTTERANCE_RULES_FILE, utterance_df, new_rule, 'utterance')
                    else: print(f"Warning: LLM suggested new utterance rule without key.")

                if isinstance(utterance.get('frame_level_directions'), list):
                    for frame in utterance['frame_level_directions']:
                         if isinstance(frame, dict) and frame.get('is_new_frame_rule') and isinstance(frame.get('new_frame_rule_definition'), dict):
                            new_frame_rule = {'key': frame.get('tag'), 'definition': frame['new_frame_rule_definition']}
                            if new_frame_rule['key']:
                                 frame_df = update_rule_file(FRAME_RULES_FILE, frame_df, new_frame_rule, 'frame')
                            else: print(f"Warning: LLM suggested new frame rule without tag.")
                
                # --- Part 3: Fetch definitions for EXISTING rules ---
                # <<< This part is no longer needed, as we force new definitions every time >>>
                # <<< We can safely remove it >>>

                # --- Part 4: Add Generation Parameters ---
                gen_params = {}
                if utterance.get('is_new_utterance_rule') and isinstance(utterance.get('new_utterance_rule_definition'), dict):
                     # Use the clamped values
                     clamped_def = utterance['new_utterance_rule_definition']
                     gen_params = {
                        "speaking_rate": clamped_def.get('speaking_rate', 15.0),
                        "pitch_std": clamped_def.get('pitch_std', 20.0)
                     }
                else:
                     # This is now an error case, as the prompt *requires* a definition
                     print(f"ERROR: LLM did not provide 'new_utterance_rule_definition' for record {current_record_index + 1}, utterance {utterance_index + 1}. Using defaults.")
                     gen_params = {"speaking_rate": 15.0, "pitch_std": 20.0}

                utterance['generation_parameters'] = gen_params

                # --- Part 5: Perform final vector calculations ---
                final_vector = calculate_final_vector(utterance, utterance_df, frame_df)
                utterance['final_zonos_vector'] = final_vector
                
                if not final_vector and utterance.get('utterance_level_direction'):
                     if utterance['new_utterance_rule_definition'].get('primary_zonos_vector_value'):
                         print(f"DEBUG: Final vector was empty but definition existed for record {current_record_index+1}, key '{utterance.get('utterance_level_direction')}'. Check calculation.")
                     else:
                         print(f"DEBUG: Final vector empty (as expected?) for record {current_record_index+1}, key '{utterance.get('utterance_level_direction')}'. Definition vector was empty.")

        else:
             print(f"Warning: Skipping calculations for record {current_record_index + 1} due to invalid/missing 'directed_utterances'.")
             if llm_output_data is None: llm_output_data = {"error": "LLM output processing failed."}

        # Construct final result
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
final_line_count = 0
if os.path.exists(OUTPUT_FILE):
     try:
          with open(OUTPUT_FILE, 'r', encoding='utf-8') as f_final:
               final_line_count = sum(1 for _ in f_final)
     except: pass
print(f"Total records now in output file '{OUTPUT_FILE}': {final_line_count}")