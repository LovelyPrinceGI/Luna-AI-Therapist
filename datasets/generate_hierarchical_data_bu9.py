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
OUTPUT_FILE = "final_hierarchical_output_v11_d20.jsonl" # <-- Make sure this is the target file
MAX_NEW_TOKENS = 8192
NUM_SAMPLES_TO_PROCESS = 20 
MAX_RETRIES = 20

# --- NEW: Define min/max for Zonos parameters ---
MIN_SPEAKING_RATE = 10.0
MAX_SPEAKING_RATE = 25.0
MIN_PITCH_STD = 20.0
MAX_PITCH_STD = 150.0 # Based on make_cond_dict comment for "expressive speech"
# --- END NEW ---


# --- 2. Load Model ---
# ... (unchanged) ...
print(f"Loading model: {MODEL_ID}...")
pipe = pipeline("text-generation", model=MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
print("Model loaded successfully!")


# --- 3. Load and Format Rules from CSV Files ---
# ... (load_rules function unchanged) ...
def load_rules():
    """Loads rules from CSVs into DataFrames and formatted strings."""
    print("Loading rules from CSV files...")
    try:
        utterance_df = pd.read_csv(UTTERANCE_RULES_FILE)
        frame_df = pd.read_csv(FRAME_RULES_FILE)

        # Format utterance rules with FULL details
        utterance_rules_list = []
        for _, row in utterance_df.iterrows():
            key = row['stage_direction_key']
            try:
                # Attempt to parse as JSON first
                zonos_vector = json.loads(str(row['primary_zonos_vector_value']).replace("'", "\""))
            except json.JSONDecodeError:
                 # Fallback for potentially malformed strings (like missing quotes)
                 # This part might need refinement depending on the actual CSV content
                zonos_vector_str_match = re.search(r'\{.*?\}', str(row['primary_zonos_vector_value']))
                if zonos_vector_str_match:
                    try:
                        zonos_vector = json.loads(zonos_vector_str_match.group(0).replace("'", "\""))
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse Zonos vector for key '{key}' even after regex. Using raw string.")
                        zonos_vector = str(row['primary_zonos_vector_value']) # Keep as string if parsing fails
                else:
                    print(f"Warning: Could not extract dict-like structure for Zonos vector for key '{key}'. Using raw value.")
                    zonos_vector = str(row['primary_zonos_vector_value'])


            full_definition = {
                "primary_zonos_vector_value": zonos_vector,
                "speaking_rate": float(row['speaking_rate']),
                "pitch_std": float(row['pitch_std'])
            }
            rule_json_string = json.dumps(full_definition)
            utterance_rules_list.append(f"- `{key}`: {rule_json_string}")
        utterance_rules_str = "\n".join(utterance_rules_list)

        # Format frame rules
        frame_rules_list = []
        for _, row in frame_df.iterrows():
            key = row['frame_level_key']
            value_raw = row['zonos_vector_value']
            # Ensure the value part is treated as a string in the prompt
            frame_rules_list.append(f"- `{key}`: {json.dumps(value_raw)}") # Use json.dumps for safety
        frame_rules_str = "\n".join(frame_rules_list)
        
        print("Rules loaded and formatted successfully.")
        return utterance_df, frame_df, utterance_rules_str, frame_rules_str

    except FileNotFoundError as e:
        print(f"Error: Could not find rule file: {e}.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during rule loading: {e}")
        exit()


# --- LUNA'S NEW FUNCTION: Automatic Rule Updater ---
# ... (update_rule_file function largely unchanged, BUT we will clamp values BEFORE calling it) ...
def update_rule_file(file_path, df, new_rule_data, rule_type):
    """
    Appends a new rule to the specified CSV file and updates the in-memory DataFrame.
    (Assumes values in new_rule_data['definition'] are already clamped/validated)
    """
    try:
        print(f"\n✨ New {'utterance' if rule_type == 'utterance' else 'frame'} rule! Updating {file_path}...")
        
        if rule_type == 'utterance':
            key = new_rule_data['key']
            definition = new_rule_data['definition'] # Assumes already clamped
            # Ensure zonos_vector is a string for CSV
            zonos_vector_csv_str = json.dumps(definition.get('primary_zonos_vector_value', {}))

            new_row_data = {
                'stage_direction_key': key,
                'primary_zonos_vector_value': zonos_vector_csv_str,
                'speaking_rate': definition.get('speaking_rate', 15.0), # Use default if missing
                'pitch_std': definition.get('pitch_std', 20.0) # Use default if missing
            }
            # Handle potential direct 'zonos_vector' key if LLM uses that instead
            if 'zonos_vector' in definition and 'primary_zonos_vector_value' not in definition:
                 new_row_data['primary_zonos_vector_value'] = json.dumps(definition['zonos_vector'])

            new_row = pd.DataFrame([new_row_data])
            new_row.to_csv(file_path, mode='a', header=not os.path.exists(file_path) or os.path.getsize(file_path) == 0, index=False)

        elif rule_type == 'frame':
            key = new_rule_data['key']
            definition = new_rule_data['definition'] # Assumes already validated
             # Ensure zonos_vector_value is a string for CSV
            zonos_vector_value_csv_str = json.dumps(definition.get('zonos_vector_value', {}))

            new_row_data = {
                'frame_level_key': key,
                'zonos_vector_value': zonos_vector_value_csv_str
            }
             # Handle potential direct 'zonos_vector' key if LLM uses that instead
            if 'zonos_vector' in definition and 'zonos_vector_value' not in definition:
                 new_row_data['zonos_vector_value'] = json.dumps(definition['zonos_vector'])

            new_row = pd.DataFrame([new_row_data])
            new_row.to_csv(file_path, mode='a', header=not os.path.exists(file_path) or os.path.getsize(file_path) == 0, index=False)

        # Update the in-memory DataFrame
        # Check if the key already exists before concatenating to avoid duplicates
        if rule_type == 'utterance' and key in df['stage_direction_key'].values:
             print(f"ℹ️ Rule '{key}' already exists in DataFrame. Skipping concat.")
             updated_df = df
        elif rule_type == 'frame' and key in df['frame_level_key'].values:
             print(f"ℹ️ Rule '{key}' already exists in DataFrame. Skipping concat.")
             updated_df = df
        else:
            updated_df = pd.concat([df, new_row], ignore_index=True)
            print(f"✅ Successfully added new rule to CSV and DataFrame: {key}")
        
        return updated_df

    except Exception as e:
        print(f"❌ Error updating rule file {file_path}: {e}")
        # print("Debug - Data that caused error:", new_rule_data) # Uncomment for debugging
        return df # Return original dataframe on failure



# --- 4. Vector Calculation Function ---
# ... (calculate_final_vector function unchanged) ...
def calculate_final_vector(utterance_output, utterance_df, frame_df, w_utterance=1.0, w_frame=0.25):
    """
    Calculates the final Zonos vector using "Luna's Weighted Fusion Formula."
    Handles both existing rules from dataframes and potentially new rules defined in the utterance_output.
    """
    try:
        final_vector = defaultdict(float)
        base_vector = {} # Initialize base_vector

        # --- Step 1: Get the Utterance Level Vector ---
        utterance_key = utterance_output.get('utterance_level_direction')
        
        # Check if it's a new rule defined directly in the output
        if utterance_output.get('is_new_utterance_rule') and 'new_utterance_rule_definition' in utterance_output:
            base_vector = utterance_output['new_utterance_rule_definition'].get('primary_zonos_vector_value', {})
            # Handle potential alternative key from LLM
            if not base_vector and 'zonos_vector' in utterance_output['new_utterance_rule_definition']:
                 base_vector = utterance_output['new_utterance_rule_definition']['zonos_vector']

        # Otherwise, look it up in the dataframe
        elif utterance_key:
            utterance_rule = utterance_df[utterance_df['stage_direction_key'] == utterance_key]
            if not utterance_rule.empty:
                base_vector_str = utterance_rule.iloc[0]['primary_zonos_vector_value']
                try:
                    # Try parsing the string as JSON
                    base_vector = json.loads(str(base_vector_str).replace("'", "\""))
                except (json.JSONDecodeError, TypeError):
                    print(f"Warning: Could not parse base vector JSON for key '{utterance_key}'. Using empty vector. Value was: {base_vector_str}")
                    base_vector = {}
            else:
                 # Check if the definition might be under 'existing_rule_definition' (added later)
                 if 'existing_rule_definition' in utterance_output:
                     base_vector = utterance_output['existing_rule_definition'].get('primary_zonos_vector_value', {})
                 else:
                    print(f"Warning: Utterance key '{utterance_key}' not found in rules DataFrame and no definition provided.")
                    base_vector = {}
        else:
             print(f"Warning: No utterance_level_direction provided.")
             base_vector = {}

        # Apply the base vector (if found)
        if isinstance(base_vector, dict):
            for emotion, value in base_vector.items():
                try:
                   final_vector[emotion] += float(value) * w_utterance
                except (ValueError, TypeError):
                   print(f"Warning: Invalid value '{value}' for emotion '{emotion}' in base vector. Skipping.")
        else:
            print(f"Warning: Base vector for utterance key '{utterance_key}' is not a dictionary: {base_vector}")


        # --- Step 2: Apply Frame Level Modifiers ---
        if 'frame_level_directions' in utterance_output:
            for frame in utterance_output['frame_level_directions']:
                frame_key = frame.get('tag')
                modifier_vector = {} # Initialize modifier_vector

                # Check if it's a new rule defined directly
                if frame.get('is_new_frame_rule') and 'new_frame_rule_definition' in frame:
                    modifier_vector = frame['new_frame_rule_definition'].get('zonos_vector_value', {})
                     # Handle potential alternative key from LLM
                    if not modifier_vector and 'zonos_vector' in frame['new_frame_rule_definition']:
                         modifier_vector = frame['new_frame_rule_definition']['zonos_vector']

                # Otherwise, look it up in the dataframe
                elif frame_key:
                    frame_rule = frame_df[frame_df['frame_level_key'] == frame_key]
                    if not frame_rule.empty:
                        modifier_str_raw = frame_rule.iloc[0]['zonos_vector_value']
                        try:
                            # Try parsing the string as JSON (might already be dict in some cases)
                            if isinstance(modifier_str_raw, str):
                                # Extract only the JSON part if necessary
                                json_match = re.search(r'\{.*?\}', modifier_str_raw)
                                if json_match:
                                    modifier_vector = json.loads(json_match.group(0).replace("'", "\""))
                                else:
                                    print(f"Warning: Could not extract dict-like structure for frame key '{frame_key}'. Value: {modifier_str_raw}")
                                    modifier_vector = {}
                            elif isinstance(modifier_str_raw, dict):
                                 modifier_vector = modifier_str_raw # Already a dict
                            else:
                                 print(f"Warning: Unexpected type for frame modifier '{frame_key}'. Type: {type(modifier_str_raw)}, Value: {modifier_str_raw}")
                                 modifier_vector = {}

                        except (json.JSONDecodeError, TypeError) as e:
                            print(f"Warning: Could not parse frame modifier JSON for key '{frame_key}'. Error: {e}. Value was: {modifier_str_raw}")
                            modifier_vector = {}
                    else:
                         # Check if the definition might be under 'existing_rule_definition' (added later)
                         if 'existing_rule_definition' in frame:
                             modifier_vector = frame['existing_rule_definition'].get('zonos_vector_value', {})
                         else:
                            print(f"Warning: Frame key '{frame_key}' not found in rules DataFrame and no definition provided.")
                            modifier_vector = {}
                else:
                     print(f"Warning: Frame direction missing 'tag'.")
                     modifier_vector = {}

                # Apply the modifier vector (if found and valid)
                if isinstance(modifier_vector, dict):
                     for emotion, mod_value_str in modifier_vector.items():
                        try:
                            # The modifier value is often like "+0.1", float handles this
                            mod_value = float(mod_value_str)
                            final_vector[emotion] += mod_value * w_frame
                        except (ValueError, TypeError):
                            print(f"Warning: Invalid modifier value '{mod_value_str}' for emotion '{emotion}' in frame key '{frame_key}'. Skipping.")
                else:
                    print(f"Warning: Frame modifier for key '{frame_key}' is not a dictionary: {modifier_vector}")


        # --- Step 3: Normalize/Clamp the final values ---
        clamped_vector = {}
        for emotion, value in final_vector.items():
            # Round to avoid excessive precision issues, e.g., 4 decimal places
            rounded_value = round(value, 4)
            clamped_vector[emotion] = max(-1.0, min(1.0, rounded_value))

        return clamped_vector

    except Exception as e:
        print(f"An error occurred during vector calculation: {e}")
        # Optionally, print the problematic utterance_output for debugging
        # import traceback
        # traceback.print_exc()
        # print("Problematic utterance data:", utterance_output)
        return {}


# --- 5. Master Prompt Templates ---
# ... (unchanged) ...
MASTER_SYSTEM_PROMPT = """You are a master Vocal Director simulating the 'emotion2vec' framework. Your task is to perform a hierarchical analysis of a therapy dialogue.

You must follow all instructions precisely and output a single, valid JSON object inside a JSON code block. You must not ask for the user to provide the JSON. You must generate it yourself based on the dialogue.

**INSTRUCTIONS:**
1.  ★★★ **UTTERANCE INTEGRITY:** The dialogue provided below contains **ONLY** the client's speech. The client's turns (utterances) are separated by a `\n\n---\n\n` delimiter. **You MUST treat each block separated by this delimiter as a *single, complete utterance*.** Do NOT split a single block into multiple `directed_utterances` entries.
2.  **ANALYSIS:** For each utterance block provided, perform a two-level analysis:
    a. **Utterance-Level:** Assign ONE overall stage direction from the "Utterance-Level Vocabulary".
    b. **Frame-Level:** Identify specific words or phrases *within* that utterance and assign one or more tags from the "Frame-Level Vocabulary".
3.  ★★★ **MAINTAIN EMOTIONAL COHERENCE:** The emotional direction of an utterance MUST be a logical progression from the *previous client utterance*. Abrupt, unrealistic emotional shifts are forbidden.

---
★★★ RULE GENERATION INSTRUCTIONS ★II
- If an existing vocabulary entry is a perfect match, USE IT.
- If NO existing vocabulary entry accurately captures the emotion, you are AUTHORIZED to CREATE A NEW ONE.
- When creating a new rule, you MUST define it completely. For a new utterance-level rule, provide the `primary_zonos_vector_value` (using ONLY valid Zonos keys), `speaking_rate` (ideal range 10-25), and `pitch_std` (ideal range 20-150). For a new frame-level rule, provide the `zonos_vector_value` (using +/- modifiers).
- The new key you create MUST be descriptive and enclosed in brackets, like `[pensive, trailing-off]`.

- ★★★ **ZONOS EMOTION VECTORS CONSTRAINT** ★★★
- When defining a `primary_zonos_vector_value`, you **MUST ONLY** use the following keys. Map complex emotions to a combination of these valid vectors. Values should generally be between -1.0 and 1.0.
    - `Happiness`: Controls cheerfulness of voice. (e.g., for emotions like Relief, Joy)
    - `Sadness`: Modifies the melancholic tone. (e.g., for Disappointment, Grief)
    - `Disgust`: Adjusts aversive qualities.
    - `Fear`: Controls anxious tone. (e.g., for Nervousness, Terror)
    - `Surprise`: Changes astonished tone.
    - `Anger`: Alters aggressive/frustrated tone. (e.g., for Frustration, Irritation)
    - `Neutral`: Balances emotion for neutrality. (e.g., for Calmness, Relief)
    - `Other`: Miscellaneous nuances.
- **The `primary_zonos_vector_value` MUST be ONLY the JSON dictionary string (e.g., `{{"Sadness": 0.8, "Neutral": 0.2}}`). DO NOT include any explanatory text, comments, or parentheses within this value.**
- When defining a `zonos_vector_value` for a NEW frame-level rule, use strings representing addition/subtraction (e.g., `"+0.2"`, `"-0.1"`). <<< MODIFIED >>> **This value MUST be ONLY the JSON dictionary string (e.g., `{{"Fear": "+0.2", "Neutral": "-0.2"}}`). DO NOT include any explanatory text.**

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
      "is_new_utterance_rule": false,
      "utterance_level_direction": "[anxious, fast]",
      "frame_level_directions": [
        {{
            "phrase": "hate me",
            "tag": "[whisper]",
            "is_new_frame_rule": false
        }}
      ]
    }},
    {{
      "utterance_text": "Well, I went to the shelter a few months ago, and some of the animals that used to greet me warmly didn’t recognize me. It felt like a punch to the gut. Since then, I’ve been avoiding going back because I can’t handle the thought of being rejected by them.",
      "is_new_utterance_rule": true,
      "utterance_level_direction": "[dejected, quiet-resignation]",
      "new_utterance_rule_definition": {{
          "primary_zonos_vector_value": {{"Sadness": 0.8, "Neutral": 0.2, "Anger": -0.1}},
          "speaking_rate": 10.0, # Clamped from original 8.0
          "pitch_std": 50.0
      }},
      "frame_level_directions": []
    }}
  ]
}}
"""

USER_PROMPT_TEMPLATE = """Here is the dialogue and the current rule sets. Please analyze the dialogue and generate the JSON output.

**EXISTING UTTERANCE-LEVEL VOCABULARY:**
{utterance_rules}

**EXISTING FRAME-LEVEL VOCABULARY:**
{frame_rules}

---

★★★ DIALOGUE TO ANALYZE (CLIENT SPEECH ONLY): ★★★
{full_dialogue}

---
Please generate the JSON output for this dialogue now. Output ONLY the JSON code block.
"""

# --- 6. Main Processing Loop ---
utterance_df, frame_df, utterance_rules_str, frame_rules_str = load_rules()

print(f"Reading input file and preparing prompts from {INPUT_FILE}...")
source_data_list = []
chats_list = []
filtered_dialogues_list = [] 
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        # Initial read to count lines if resuming
        total_lines = sum(1 for line in infile)
        infile.seek(0) # Reset file pointer

        start_index = 0
        if os.path.exists(OUTPUT_FILE):
             try:
                 with open(OUTPUT_FILE, 'r', encoding='utf-8') as outfile_check:
                      start_index = sum(1 for line in outfile_check)
                      print(f"Output file exists. Resuming from record #{start_index + 1}...")
             except Exception as e:
                  print(f"Warning: Could not read existing output file to determine resume point. Starting from beginning. Error: {e}")
        else:
             print("Output file does not exist. Starting a new processing job.")

        # Read only necessary lines if resuming
        for i, line in enumerate(infile):
             if i < start_index:
                  continue # Skip lines already processed

             if NUM_SAMPLES_TO_PROCESS is not None and (i - start_index) >= NUM_SAMPLES_TO_PROCESS:
                 print(f"--- Reached NUM_SAMPLES_TO_PROCESS limit ({NUM_SAMPLES_TO_PROCESS}). Stopping prompt preparation. ---")
                 break # Stop reading further input lines

             try:
                 data = json.loads(line.strip())
                 source_data_list.append(data) # Keep original data for final output
                 dialogue = data.get("dialogue", "")

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
                         # --- Update prompts dynamically with latest rules ---
                         # Reload rules inside loop IF a new rule was added in previous iteration?
                         # For simplicity, we'll update the strings here *before* creating the chat.
                         # This assumes update_rule_file correctly updates the DFs.

                         # Re-format utterance rules string from the potentially updated DF
                         current_utterance_rules_list = []
                         for _, row in utterance_df.iterrows():
                              key = row['stage_direction_key']
                              try:
                                   zonos_vector = json.loads(str(row['primary_zonos_vector_value']).replace("'", "\""))
                              except: zonos_vector = str(row['primary_zonos_vector_value']) # Fallback
                              full_definition = {
                                   "primary_zonos_vector_value": zonos_vector,
                                   "speaking_rate": float(row['speaking_rate']),
                                   "pitch_std": float(row['pitch_std'])
                              }
                              current_utterance_rules_list.append(f"- `{key}`: {json.dumps(full_definition)}")
                         current_utterance_rules_str = "\n".join(current_utterance_rules_list)

                         # Re-format frame rules string
                         current_frame_rules_list = []
                         for _, row in frame_df.iterrows():
                             key = row['frame_level_key']
                             value_raw = row['zonos_vector_value']
                             current_frame_rules_list.append(f"- `{key}`: {json.dumps(value_raw)}")
                         current_frame_rules_str = "\n".join(current_frame_rules_list)
                         # --- End Dynamic Prompt Update ---


                         chat_messages = [
                             {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                             {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                                 utterance_rules=current_utterance_rules_str, # Use current rules
                                 frame_rules=current_frame_rules_str,       # Use current rules
                                 full_dialogue=filtered_dialogue
                             )}
                         ]
                         chats_list.append(chat_messages)
                         filtered_dialogues_list.append(filtered_dialogue)
                     else:
                         # No client dialogue found after filtering
                         source_data_list.pop() # Remove the source data as we won't process it
                         print(f"Warning: No client utterances found in record {i+1}. Skipping.")

                 else:
                     # Original dialogue was empty
                     source_data_list.pop() # Remove the source data
                     print(f"Warning: Empty dialogue field in record {i+1}. Skipping.")

             except json.JSONDecodeError:
                 print(f"Warning: Could not decode JSON on input line {i+1}. Skipping.")
                 # If source_data_list was appended before error, remove it
                 if len(source_data_list) > len(chats_list): source_data_list.pop()


except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()
except Exception as e:
     print(f"An unexpected error occurred during input file reading: {e}")
     # import traceback
     # traceback.print_exc() # Uncomment for detailed traceback
     exit()


print(f"Prepared {len(chats_list)} chats to process for this run.")
if not chats_list:
     print("No chats remaining or prepared. Exiting.")
     exit()

print(f"Starting processing from record #{start_index + 1}...")
processing_start_time = time.time()

# We only need to iterate through the chats we prepared for *this run*
with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
    for i, chat in enumerate(tqdm(chats_list, desc="Processing Dialogues")):
        current_record_index = start_index + i # Keep track of original index
        llm_output_data = None
        generated_text = ""

        for attempt in range(MAX_RETRIES):
            try:
                # Call pipe with the chat list
                outputs = pipe(chat, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=pipe.tokenizer.eos_token_id, num_return_sequences=1) # Ensure only 1 sequence

                # Get the assistant's reply
                # The structure might vary slightly, adjust if needed
                if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
                    # Handle case where pipe returns the full chat history
                    if isinstance(outputs[0]['generated_text'], list):
                         generated_text = outputs[0]['generated_text'][-1]['content'].strip()
                    # Handle case where pipe returns only the generated part
                    elif isinstance(outputs[0]['generated_text'], str):
                         # Need to find the start of the assistant's actual reply
                         user_prompt_end = chat[-1]['content']
                         if user_prompt_end in outputs[0]['generated_text']:
                              generated_text = outputs[0]['generated_text'].split(user_prompt_end, 1)[-1].strip()
                         else: # Fallback if split fails
                              generated_text = outputs[0]['generated_text'].strip()

                    else:
                         raise ValueError(f"Unexpected format for generated_text: {type(outputs[0]['generated_text'])}")

                else:
                    raise ValueError(f"Unexpected output format from pipeline: {outputs}")


                json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                if json_match:
                    json_string = json_match.group(1)
                    try:
                        llm_output_data = json.loads(json_string)
                        # Basic validation
                        if 'directed_utterances' not in llm_output_data or not isinstance(llm_output_data['directed_utterances'], list):
                             raise ValueError("JSON missing 'directed_utterances' list.")
                        break # Success
                    except json.JSONDecodeError as json_e:
                        print(f"\nJSON Decode Error (Attempt {attempt + 1}): {json_e}")
                        print(f"Invalid JSON string: {json_string[:500]}...") # Print beginning of invalid JSON
                        # Keep trying, maybe next attempt works
                else:
                    print(f"\nWarning (Attempt {attempt + 1}): Could not find JSON block in output.")
                    print(f"Raw output sample: {generated_text[:]}...")
                    # Let it retry

                # If loop finishes without break, raise error for this attempt
                if attempt == MAX_RETRIES - 1 and not llm_output_data:
                     raise ValueError("Could not find/parse valid JSON after multiple retries.")


            except Exception as e:
                print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for record {current_record_index + 1}. Error: {e}")
                # import traceback # Uncomment for debugging specific errors
                # traceback.print_exc() # Uncomment for debugging specific errors
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2) # Increased sleep time
                else:
                    print(f"Max retries reached for record {current_record_index + 1}. Saving placeholder.")
                    llm_output_data = {"error": f"Failed after {MAX_RETRIES} retries.", "raw_output": generated_text} # Save raw output on final failure


        # --- LUNA'S DYNAMIC RULE & CALCULATION LOGIC (UPGRADED + CLAMPING) ---
        if llm_output_data and isinstance(llm_output_data.get('directed_utterances'), list):
            for utterance_index, utterance in enumerate(llm_output_data['directed_utterances']):
                # Ensure utterance is a dictionary
                if not isinstance(utterance, dict):
                    print(f"Warning: Skipping invalid utterance data (not a dict) in record {current_record_index + 1}, utterance {utterance_index + 1}.")
                    continue # Skip this utterance


                # Part 1: Handle NEW utterance rules (CLAMPING ADDED HERE)
                if utterance.get('is_new_utterance_rule') and isinstance(utterance.get('new_utterance_rule_definition'), dict):
                    new_def = utterance['new_utterance_rule_definition']
                    
                    # <<< NEW CLAMPING LOGIC for Utterance Rules >>>
                    original_rate = new_def.get('speaking_rate')
                    original_pitch = new_def.get('pitch_std')
                    
                    if original_rate is not None:
                         try:
                             rate_float = float(original_rate)
                             clamped_rate = max(MIN_SPEAKING_RATE, min(rate_float, MAX_SPEAKING_RATE))
                             if clamped_rate != rate_float:
                                 print(f" INFO: Clamped LLM suggested speaking_rate from {rate_float} to {clamped_rate} for new rule '{utterance.get('utterance_level_direction')}'.")
                             new_def['speaking_rate'] = clamped_rate
                         except (ValueError, TypeError):
                              print(f"Warning: Invalid speaking_rate '{original_rate}' suggested by LLM for new rule. Using default 15.0.")
                              new_def['speaking_rate'] = 15.0
                    else:
                         new_def['speaking_rate'] = 15.0 # Assign default if missing
                         
                    if original_pitch is not None:
                         try:
                             pitch_float = float(original_pitch)
                             clamped_pitch = max(MIN_PITCH_STD, min(pitch_float, MAX_PITCH_STD))
                             if clamped_pitch != pitch_float:
                                  print(f" INFO: Clamped LLM suggested pitch_std from {pitch_float} to {clamped_pitch} for new rule '{utterance.get('utterance_level_direction')}'.")
                             new_def['pitch_std'] = clamped_pitch
                         except (ValueError, TypeError):
                              print(f"Warning: Invalid pitch_std '{original_pitch}' suggested by LLM for new rule. Using default 20.0.")
                              new_def['pitch_std'] = 20.0
                    else:
                         new_def['pitch_std'] = 20.0 # Assign default if missing
                    # <<< END CLAMPING LOGIC >>>

                    new_rule = {
                        'key': utterance.get('utterance_level_direction'),
                        'definition': new_def # Use the potentially clamped definition
                    }
                    if new_rule['key']: # Only update if key is valid
                         # Update DF and CSV (function handles check for existence)
                         utterance_df = update_rule_file(UTTERANCE_RULES_FILE, utterance_df, new_rule, 'utterance')
                    else:
                         print(f"Warning: LLM suggested a new utterance rule but provided no key.")


                # Part 2: Handle NEW frame rules (No clamping needed for modifiers)
                if isinstance(utterance.get('frame_level_directions'), list):
                    for frame in utterance['frame_level_directions']:
                         if isinstance(frame, dict) and frame.get('is_new_frame_rule') and isinstance(frame.get('new_frame_rule_definition'), dict):
                            new_frame_rule = {
                                'key': frame.get('tag'),
                                'definition': frame['new_frame_rule_definition']
                            }
                            if new_frame_rule['key']: # Only update if key is valid
                                 # Update DF and CSV (function handles check for existence)
                                 frame_df = update_rule_file(FRAME_RULES_FILE, frame_df, new_frame_rule, 'frame')
                            else:
                                 print(f"Warning: LLM suggested a new frame rule but provided no tag.")


                # Part 3: Fetch definitions for EXISTING rules (unchanged logic)
                # For Utterance Level
                if not utterance.get('is_new_utterance_rule'):
                    key = utterance.get('utterance_level_direction')
                    if key:
                        rule = utterance_df[utterance_df['stage_direction_key'] == key]
                        if not rule.empty:
                            rule_data = rule.iloc[0].to_dict() # Convert to dict for easier access
                            try:
                                # Prioritize parsing vector string
                                vector_str = str(rule_data.get('primary_zonos_vector_value', '{}'))
                                vector_match = re.search(r'\{.*?\}', vector_str) # More robust regex
                                vector_val = json.loads(vector_match.group(0).replace("'", "\"")) if vector_match else {}

                            except (json.JSONDecodeError, AttributeError, TypeError):
                                print(f"Warning: Could not parse existing utterance vector for key '{key}'. Using empty. Value: {rule_data.get('primary_zonos_vector_value')}")
                                vector_val = {}
                            
                            utterance['existing_rule_definition'] = {
                                "primary_zonos_vector_value": vector_val,
                                "speaking_rate": float(rule_data.get('speaking_rate', 15.0)), # Add default
                                "pitch_std": float(rule_data.get('pitch_std', 20.0))          # Add default
                            }

                # For Frame Level
                if isinstance(utterance.get('frame_level_directions'), list):
                    for frame in utterance['frame_level_directions']:
                         if isinstance(frame, dict) and not frame.get('is_new_frame_rule'):
                            tag = frame.get('tag')
                            if tag:
                                rule = frame_df[frame_df['frame_level_key'] == tag]
                                if not rule.empty:
                                    rule_data = rule.iloc[0].to_dict()
                                    try:
                                         vector_str = str(rule_data.get('zonos_vector_value', '{}'))
                                         vector_match = re.search(r'\{.*?\}', vector_str)
                                         vector_val = json.loads(vector_match.group(0).replace("'", "\"")) if vector_match else {}
                                    except (json.JSONDecodeError, AttributeError, TypeError):
                                         print(f"Warning: Could not parse existing frame vector for tag '{tag}'. Using empty. Value: {rule_data.get('zonos_vector_value')}")
                                         vector_val = {}
                                    frame['existing_rule_definition'] = {"zonos_vector_value": vector_val}


                # Part 4: Add Generation Parameters (using clamped values for new rules)
                # Retrieve parameters based on whether it's new or existing
                gen_params = {}
                if utterance.get('is_new_utterance_rule') and 'new_utterance_rule_definition' in utterance:
                     # Use the already clamped values from Part 1
                     clamped_def = utterance['new_utterance_rule_definition']
                     gen_params = {
                        "speaking_rate": clamped_def.get('speaking_rate', 15.0),
                        "pitch_std": clamped_def.get('pitch_std', 20.0)
                     }
                elif 'existing_rule_definition' in utterance:
                     # Use values fetched from CSV/DF
                     existing_def = utterance['existing_rule_definition']
                     gen_params = {
                         "speaking_rate": existing_def.get('speaking_rate', 15.0),
                         "pitch_std": existing_def.get('pitch_std', 20.0)
                     }
                else:
                     # Fallback if definition couldn't be found/created
                     print(f"Warning: Could not determine generation parameters for utterance {utterance_index + 1} in record {current_record_index + 1}. Using defaults.")
                     gen_params = {"speaking_rate": 15.0, "pitch_std": 20.0}

                utterance['generation_parameters'] = gen_params


                # Part 5: Perform final vector calculations (using updated DFs)
                final_vector = calculate_final_vector(utterance, utterance_df, frame_df)
                utterance['final_zonos_vector'] = final_vector
                
        else:
             # Handle cases where LLM failed or output was invalid
             print(f"Warning: 'directed_utterances' key missing or invalid in LLM output for record {current_record_index + 1}. Skipping calculations for this record.")
             # Ensure llm_output_data exists even if it's just the error placeholder
             if llm_output_data is None: llm_output_data = {"error": "LLM output processing failed before calculations."}


        # --- END OF DYNAMIC LOGIC ---

        # Construct final result, ensuring source_data index matches chat index
        result_data = {
            "original_dialogue_full": source_data_list[i], # Use the source_data corresponding to the current chat index
            "filtered_client_dialogue": filtered_dialogues_list[i],
            "llm_output": llm_output_data
        }
        outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')
        outfile.flush() # Ensure data is written immediately

processing_end_time = time.time()
print("\n--- Processing Complete! ---")
total_processed_this_run = len(chats_list)
print(f"Total records processed in this session: {total_processed_this_run}")
if total_processed_this_run > 0:
     print(f"Processing time: {processing_end_time - processing_start_time:.2f} seconds ({ (processing_end_time - processing_start_time) / total_processed_this_run :.2f} sec/record)")
print(f"Total records in output file '{OUTPUT_FILE}': {start_index + total_processed_this_run}")