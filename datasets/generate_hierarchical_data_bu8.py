import torch
from transformers import pipeline
import json
import time
from tqdm import tqdm
import pandas as pd
import re
from collections import defaultdict
import os # Import os for file path handling

# --- 1. Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
UTTERANCE_RULES_FILE = "utterance_level_rules.csv"
FRAME_RULES_FILE = "frame_level_rules.csv"
INPUT_FILE = "cactus_huggingface_train.jsonl"
OUTPUT_FILE = "final_hierarchical_output_v11_d12.jsonl" # <-- V11 with 12 dialogues
MAX_NEW_TOKENS = 2048
NUM_SAMPLES_TO_PROCESS = 20 # Set to None for the full run
MAX_RETRIES = 3

# --- 2. Load Model ---
print(f"Loading model: {MODEL_ID}...")
# Make sure to include device_map="auto" to use available GPUs
pipe = pipeline("text-generation", model=MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
print("Model loaded successfully!")

# --- 3. Load and Format Rules from CSV Files ---
def load_rules():
    """Loads rules from CSVs into DataFrames and formatted strings."""
    print("Loading rules from CSV files...")
    try:
        utterance_df = pd.read_csv(UTTERANCE_RULES_FILE)
        frame_df = pd.read_csv(FRAME_RULES_FILE)

        # --- LUNA'S UPDATE: Format utterance rules with FULL details ---
        utterance_rules_list = []
        for _, row in utterance_df.iterrows():
            key = row['stage_direction_key']
            # Safely parse the zonos vector string into a Python dictionary
            try:
                zonos_vector_str = re.match(r'(\{.*\})', str(row['primary_zonos_vector_value'])).group(1)
                zonos_vector = json.loads(zonos_vector_str)
            except (AttributeError, json.JSONDecodeError):
                print(f"Warning: Could not parse Zonos vector for key '{key}'. Using raw value.")
                zonos_vector = str(row['primary_zonos_vector_value'])

            # Create the full definition dictionary
            full_definition = {
                "primary_zonos_vector_value": zonos_vector,
                "speaking_rate": float(row['speaking_rate']),
                "pitch_std": float(row['pitch_std'])
            }
            # Convert the full definition back to a compact JSON string for the prompt
            rule_json_string = json.dumps(full_definition)
            utterance_rules_list.append(f"- `{key}`: {rule_json_string}")
        utterance_rules_str = "\n".join(utterance_rules_list)
        # --- END OF UPDATE ---

        # Format frame rules for the prompt (can remain simple)
        frame_rules_list = []
        for _, row in frame_df.iterrows():
            key = row['frame_level_key']
            value_raw = row['zonos_vector_value']
            frame_rules_list.append(f"- `{key}`: {value_raw}")
        frame_rules_str = "\n".join(frame_rules_list)
        
        print("Rules loaded and formatted successfully.")
        return utterance_df, frame_df, utterance_rules_str, frame_rules_str

    except FileNotFoundError as e:
        print(f"Error: Could not find rule file: {e}. Please ensure CSV files are in the same directory.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during rule loading: {e}")
        exit()

# --- LUNA'S NEW FUNCTION: Automatic Rule Updater ---
def update_rule_file(file_path, df, new_rule_data, rule_type):
    """
    Appends a new rule to the specified CSV file and updates the in-memory DataFrame.
    """
    try:
        print(f"\n✨ New {'utterance' if rule_type == 'utterance' else 'frame'} rule detected! Updating {file_path}...")
        
        if rule_type == 'utterance':
            key = new_rule_data['key']
            definition = new_rule_data['definition']
            # Create a new DataFrame for the new rule
            new_row = pd.DataFrame([{
                'stage_direction_key': key,
                'primary_zonos_vector_value': json.dumps(definition['zonos_vector']),
                'speaking_rate': definition['generation_parameters']['speaking_rate'],
                'pitch_std': definition['generation_parameters']['pitch_std']
            }])
            # Append to the CSV file
            new_row.to_csv(file_path, mode='a', header=False, index=False)
            
        elif rule_type == 'frame':
            key = new_rule_data['key']
            definition = new_rule_data['definition']
            new_row = pd.DataFrame([{
                'frame_level_key': key,
                'zonos_vector_value': json.dumps(definition['zonos_vector'])
            }])
            new_row.to_csv(file_path, mode='a', header=False, index=False)

        # Update the in-memory DataFrame to make the new rule available immediately
        updated_df = pd.concat([df, new_row], ignore_index=True)
        print(f"✅ Successfully added new rule: {key}")
        return updated_df

    except Exception as e:
        print(f"❌ Error updating rule file {file_path}: {e}")
        return df # Return original dataframe on failure

# --- 4. Vector Calculation Function (Luna's Weighted Fusion Formula) ---

def calculate_final_vector(llm_output, utterance_df, frame_df, w_utterance=1.0, w_frame=0.25):
    """
    Calculates the final Zonos vector using "Luna's Weighted Fusion Formula."
    ... (This function remains unchanged from V9/V10) ...
    """
    try:
        # Use defaultdict to easily handle addition of new emotion keys
        final_vector = defaultdict(float)
        
        # --- Step 1: Apply the Utterance Level Vector ---
        utterance_key = llm_output['utterance_level_direction']
        
        # Find the corresponding rule in the utterance dataframe
        utterance_rule = utterance_df[utterance_df['stage_direction_key'] == utterance_key]
        if not utterance_rule.empty:
            # --- LUNA'S V10 FIX for new rules that aren't in the DF yet ---
            if 'primary_zonos_vector_value' in utterance_rule.iloc[0]:
                base_vector_str = utterance_rule.iloc[0]['primary_zonos_vector_value']
            # Handle case where the rule was *just* added by the LLM
            elif 'new_utterance_rule_definition' in llm_output and 'primary_zonos_vector_value' in llm_output['new_utterance_rule_definition']:
                 base_vector_str = json.dumps(llm_output['new_utterance_rule_definition']['primary_zonos_vector_value'])
            else:
                print(f"Warning: Could not find vector definition for new key '{utterance_key}'.")
                base_vector_str = "{}"
            # --- END OF V10 FIX ---

            base_vector = json.loads(base_vector_str)
            
            for emotion, value in base_vector.items():
                final_vector[emotion] += value * w_utterance
        else:
             # --- LUNA'S V11 FIX (copy from V10): Handle new rules that aren't in the DF *at all* ---
            if 'new_utterance_rule_definition' in llm_output and 'primary_zonos_vector_value' in llm_output['new_utterance_rule_definition']:
                 base_vector_str = json.dumps(llm_output['new_utterance_rule_definition']['primary_zonos_vector_value'])
                 base_vector = json.loads(base_vector_str)
                 for emotion, value in base_vector.items():
                     final_vector[emotion] += value * w_utterance
            else:
                print(f"Warning: Utterance key '{utterance_key}' not found in rules and no new definition provided.")
            # --- END OF V11 FIX ---


        # --- Step 2: Apply Frame Level Modifiers ---
        if 'frame_level_directions' in llm_output:
            for frame in llm_output['frame_level_directions']:
                frame_key = frame['tag']
                frame_rule = frame_df[frame_df['frame_level_key'] == frame_key]
                
                if not frame_rule.empty:
                    modifier_str_raw = frame_rule.iloc[0]['zonos_vector_value']
                    # Extract the JSON part from the string
                    json_part = re.match(r'(\{.*\})', modifier_str_raw)
                    if json_part:
                        modifier_vector = json.loads(json_part.group(1))
                        for emotion, mod_value in modifier_vector.items():
                            # The modifier value is a string like "+0.1", float() handles it correctly
                            final_vector[emotion] += float(mod_value) * w_frame
                # --- LUNA'S V10 FIX for new frame rules ---
                elif frame.get('is_new_frame_rule') and 'new_frame_rule_definition' in frame:
                     modifier_str_raw = json.dumps(frame['new_frame_rule_definition']['zonos_vector_value'])
                     json_part = re.match(r'(\{.*\})', modifier_str_raw)
                     if json_part:
                        modifier_vector = json.loads(json_part.group(1))
                        for emotion, mod_value in modifier_vector.items():
                            final_vector[emotion] += float(mod_value) * w_frame
                # --- END OF V10 FIX ---
                else:
                    print(f"Warning: Frame key '{frame_key}' not found in rules.")

        # --- Step 3: Normalize/Clamp the final values (e.g., between -1.0 and 1.0) ---
        clamped_vector = {}
        for emotion, value in final_vector.items():
            clamped_vector[emotion] = max(-1.0, min(1.0, value))
            
        return clamped_vector

    except Exception as e:
        print(f"An error occurred during vector calculation: {e}")
        return {}

# --- 5. Master Prompt Templates (LUNA'S V10 PRE-FILTERING FIX) ---
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
- When creating a new rule, you MUST define it completely. For a new utterance-level rule, provide the `zonos_vector`, `speaking_rate`, and `pitch_std`. For a new frame-level rule, provide the `zonos_vector_value`.
- The new key you create MUST be descriptive and enclosed in brackets, like `[pensive, trailing-off]`.

- ★★★ **ZONOS EMOTION VECTORS CONSTRAINT** ★★★
- When defining a `zonos_vector`, you **MUST ONLY** use the following keys. Map complex emotions to a combination of these valid vectors.
    - `Happiness`: Controls cheerfulness of voice. (e.g., for emotions like Relief, Joy)
    - `Sadness`: Modifies the melancholic tone. (e.g., for Disappointment, Grief)
    - `Disgust`: Adjusts aversive qualities.
    - `Fear`: Controls anxious tone. (e.g., for Nervousness, Terror)
    - `Surprise`: Changes astonished tone.
    - `Anger`: Alters aggressive/frustrated tone. (e.g., for Frustration, Irritation)
    - `Neutral`: Balances emotion for neutrality. (e.g., for Calmness, Relief)
    - `Other`: Miscellaneous nuances.

- **Example of Correct Mapping:**
    - Emotion: "Relieved" -> Should be mapped to a combination like `{{"Happiness": 0.6, "Neutral": 0.7}}`.
    - Emotion: "Frustrated" -> Should be mapped to `{{"Anger": 0.7, "Sadness": 0.2}}`.

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
          "speaking_rate": 8,
          "pitch_std": 50
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
filtered_dialogues_list = [] # <-- LUNA'S V11 FIX: Create a new list for filtered dialogues
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            source_data_list.append(data)
            dialogue = data.get("dialogue", "")
            
            if dialogue:
                # --- LUNA'S V10 PYTHON PRE-FILTER ---
                client_utterances = []
                current_utterance_lines = []
                
                for line in dialogue.split('\n'):
                    line_stripped = line.strip()
                    
                    if line_stripped.startswith("Client:"):
                        # If we have lines from a previous utterance, join and save them
                        if current_utterance_lines:
                            client_utterances.append(" ".join(current_utterance_lines))
                        # Start the new utterance, stripping the "Client: " prefix
                        current_utterance_lines = [line_stripped[len("Client:"):].strip()]
                    
                    elif line_stripped.startswith("Counselor:"):
                        # If we hit a counselor, save the last client utterance and reset
                        if current_utterance_lines:
                            client_utterances.append(" ".join(current_utterance_lines))
                        current_utterance_lines = [] # Reset
                        
                    elif current_utterance_lines and line_stripped: # This is a continuation line for the *current* client utterance
                        current_utterance_lines.append(line_stripped)
                
                # Add the very last utterance if it exists
                if current_utterance_lines:
                    client_utterances.append(" ".join(current_utterance_lines))
                
                # Join all client utterances with a clear delimiter for the LLM
                filtered_dialogue = "\n\n---\n\n".join(client_utterances)
                # --- END OF V10 FILTER ---
                
                if filtered_dialogue: # Check if we actually found any client lines
                    chat_messages = [
                        {"role": "system", "content": MASTER_SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                            utterance_rules=utterance_rules_str,
                            frame_rules=frame_rules_str,
                            full_dialogue=filtered_dialogue # V10: Use the filtered dialogue
                        )}
                    ]
                    chats_list.append(chat_messages)
                    filtered_dialogues_list.append(filtered_dialogue) # <-- LUNA'S V11 FIX: Add the correct dialogue to the list
                else:
                    chats_list.append(None) # No client dialogue found
                    filtered_dialogues_list.append(None) # <-- LUNA'S V11 FIX: Add placeholder
            else:
                chats_list.append(None) # Add None placeholder for empty dialogues
                filtered_dialogues_list.append(None) # <-- LUNA'S V11 FIX: Add placeholder

except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()

if NUM_SAMPLES_TO_PROCESS is not None:
    source_data_list = source_data_list[:NUM_SAMPLES_TO_PROCESS]
    chats_list = chats_list[:NUM_SAMPLES_TO_PROCESS]
    filtered_dialogues_list = filtered_dialogues_list[:NUM_SAMPLES_TO_PROCESS] # <-- LUNA'S V11 FIX: Slice this list too
    print(f"--- Limiting to the first {NUM_SAMPLES_TO_PROCESS} samples for this run. ---")

print(f"Prepared {len(chats_list)} chats to process.")
print(f"Starting processing...")
start_time = time.time()

processed_count = 0
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        processed_count = sum(1 for line in f)
    print(f"Resuming from record #{processed_count + 1}...")
else:
    print("Starting a new processing job.")

remaining_chats = chats_list[processed_count:] 
remaining_source_data = source_data_list[processed_count:]
remaining_filtered_dialogues = filtered_dialogues_list[processed_count:] # <-- LUNA'S V11 FIX: Get the remaining slice

with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
    if not remaining_chats:
        print("All prompts have already been processed.")
    else:
        for i, chat in enumerate(tqdm(remaining_chats, desc="Processing Dialogues")): 
            llm_output_data = None
            generated_text = ""
            for attempt in range(MAX_RETRIES):
                try:
                    if not chat: # Handle empty dialogues
                        llm_output_data = {"error": "Empty dialogue."}
                        break
                    
                    # Call pipe with the chat list
                    outputs = pipe(chat, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=pipe.tokenizer.eos_token_id)
                    
                    # Get the assistant's reply from the chat history
                    raw_output_chat = outputs[0]['generated_text']
                    generated_text = raw_output_chat[-1]['content'].strip() # Get the last message (assistant's reply)

                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', generated_text, re.DOTALL)
                    if json_match:
                        json_string = json_match.group(1)
                        llm_output_data = json.loads(json_string)
                        break
                    else:
                        print(f"\nRaw output from LLM (Attempt {attempt + 1}): {generated_text}")
                        raise ValueError("Could not find a valid JSON code block in the assistant's reply.")

                except Exception as e:
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed. Error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(1)
                    else:
                        print("Max retries reached. Saving placeholder.")
                        llm_output_data = {"error": f"Failed after {MAX_RETRIES} retries.", "raw_output": generated_text}

            # --- LUNA'S DYNAMIC RULE & CALCULATION LOGIC (UPGRADED) ---
            if llm_output_data and 'directed_utterances' in llm_output_data:
                for utterance in llm_output_data['directed_utterances']:
                    # Part 1: Handle NEW rules (and update CSVs)
                    if utterance.get('is_new_utterance_rule') and 'new_utterance_rule_definition' in utterance:
                        new_rule = {
                            'key': utterance['utterance_level_direction'],
                            'definition': utterance['new_utterance_rule_definition']
                        }
                        utterance_df = update_rule_file(UTTERANCE_RULES_FILE, utterance_df, new_rule, 'utterance')
                    
                    if 'frame_level_directions' in utterance:
                        for frame in utterance['frame_level_directions']:
                            if frame.get('is_new_frame_rule') and 'new_frame_rule_definition' in frame:
                                new_frame_rule = {'key': frame['tag'], 'definition': frame['new_frame_rule_definition']}
                                frame_df = update_rule_file(FRAME_RULES_FILE, frame_df, new_frame_rule, 'frame')

                    # --- LUNA'S NEW ADDITION: Fetch and add definitions for EXISTING rules ---
                    # For Utterance Level
                    if not utterance.get('is_new_utterance_rule'):
                        key = utterance.get('utterance_level_direction')
                        if key:
                            rule = utterance_df[utterance_df['stage_direction_key'] == key]
                            if not rule.empty:
                                rule_data = rule.iloc[0]
                                try:
                                    vector_val = json.loads(re.match(r'(\{.*\})', str(rule_data['primary_zonos_vector_value'])).group(1))
                                except:
                                    vector_val = {}
                                utterance['existing_rule_definition'] = {
                                    "primary_zonos_vector_value": vector_val,
                                    "speaking_rate": float(rule_data['speaking_rate']),
                                    "pitch_std": float(rule_data['pitch_std'])
                                }
                    
                    # For Frame Level
                    if 'frame_level_directions' in utterance:
                        for frame in utterance['frame_level_directions']:
                            if not frame.get('is_new_frame_rule'):
                                tag = frame.get('tag')
                                if tag:
                                    rule = frame_df[frame_df['frame_level_key'] == tag]
                                    if not rule.empty:
                                        rule_data = rule.iloc[0]
                                        try:
                                            vector_val = json.loads(re.match(r'(\{.*\})', str(rule_data['zonos_vector_value'])).group(1))
                                        except:
                                            vector_val = {}
                                        frame['existing_rule_definition'] = {"zonos_vector_value": vector_val}
                    # --- END OF NEW ADDITION ---

                    # Part 3: Perform final calculations
                    final_vector = calculate_final_vector(utterance, utterance_df, frame_df)
                    utterance['final_zonos_vector'] = final_vector
                    
                    # --- LUNA'S V10 FIX for Generation Parameters on NEW rules ---
                    utterance_key = utterance.get('utterance_level_direction')
                    rule = utterance_df[utterance_df['stage_direction_key'] == utterance_key]
                    if not rule.empty:
                        utterance['generation_parameters'] = {
                            "speaking_rate": float(rule.iloc[0]['speaking_rate']),
                            "pitch_std": float(rule.iloc[0]['pitch_std'])
                        }
                    # Handle case where the rule was *just* added by the LLM
                    elif utterance.get('is_new_utterance_rule') and 'new_utterance_rule_definition' in utterance:
                         utterance['generation_parameters'] = {
                            "speaking_rate": float(utterance['new_utterance_rule_definition']['speaking_rate']),
                            "pitch_std": float(utterance['new_utterance_rule_definition']['pitch_std'])
                        }
                    # --- END OF V10 FIX ---

            # --- END OF DYNAMIC LOGIC ---

            result_data = {
                "original_dialogue_full": remaining_source_data[i], # V10: Keep the original full dialogue for reference
                "filtered_client_dialogue": remaining_filtered_dialogues[i], # <-- LUNA'S V11 FIX: Get the correct dialogue from the list
                "llm_output": llm_output_data
            }
            outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')

end_time = time.time()
print("\n--- Processing Complete! ---")
print(f"Total records processed in this session: {len(remaining_chats)}")
print(f"Processing time: {end_time - start_time:.2f} seconds")