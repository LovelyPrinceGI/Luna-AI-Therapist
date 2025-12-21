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
OUTPUT_FILE = "final_hierarchical_output_v6.jsonl" 
MAX_NEW_TOKENS = 2048
NUM_SAMPLES_TO_PROCESS = 5 # Set to None for the full run
MAX_RETRIES = 3

# --- 2. Load Model ---
print(f"Loading model: {MODEL_ID}...")
# Make sure to include device_map="auto" to use available GPUs
pipe = pipeline("text-generation", model=MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
print("Model loaded successfully!")

# --- 3. Load and Format Rules from CSV Files ---
print("Loading rules from CSV files...")
try:
    utterance_df = pd.read_csv(UTTERANCE_RULES_FILE)
    utterance_rules_list = []
    # --- LUNA'S UPDATE START ---
    # Loop through each row to build the detailed rule string including generation parameters
    for index, row in utterance_df.iterrows():
        key = row['stage_direction_key']
        # --- LUNA'S UPDATE TO FIX PARSING ---
        value_raw = str(row['primary_zonos_vector_value']) # Convert to string just in case
        json_part = re.match(r'(\{.*\})', value_raw)
        
        if not json_part:
            print(f"Warning: Could not extract a valid JSON object for key '{key}'. Skipping.")
            continue
        
        zonos_vector_str = json_part.group(1)
        # --- END OF UPDATE ---        
        # Safely parse the zonos vector string into a Python dictionary
        try:
            zonos_vector = json.loads(zonos_vector_str)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse Zonos vector for key '{key}'. Skipping.")
            continue
        
        # Create the generation parameters dictionary
        generation_parameters = {
            "speaking_rate": float(row['speaking_rate']),
            "pitch_std": float(row['pitch_std'])
        }
        
        # Combine everything into the final structure
        final_rule_structure = {
            "zonos_vector": zonos_vector,
            "generation_parameters": generation_parameters
        }
        
        # Convert the final structure back to a compact JSON string and format the line
        rule_json_string = json.dumps(final_rule_structure)
        utterance_rules_list.append(f"- `{key}`: {rule_json_string}")
    # --- LUNA'S UPDATE END ---
    utterance_rules_str = "\n".join(utterance_rules_list)

    frame_df = pd.read_csv(FRAME_RULES_FILE)
    frame_rules_list = []
    for index, row in frame_df.iterrows():
        key = row['frame_level_key']
        value_raw = row['zonos_vector_value']
        json_part = re.match(r'(\{.*\})', value_raw)
        if json_part:
            value = json_part.group(1)
            frame_rules_list.append(f"- `{key}`: {value}")
    frame_rules_str = "\n".join(frame_rules_list)
    print("Rules loaded and formatted successfully.")
    
    # Optional: Print to verify the new format
    # print("\n--- Formatted Utterance Rules ---")
    # print(utterance_rules_str)
    # print("\n")

except FileNotFoundError as e:
    print(f"Error: Could not find rule file: {e}. Please ensure CSV files are in the same directory.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
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

    This method combines a primary utterance-level emotion vector with one or more
    frame-level modifier vectors. It treats the utterance vector as the foundational
    emotion and applies the frame vectors as subtle adjustments, controlled by weights.

    **Formula:**
    V_final = (w_utterance * V_utterance) + sum(w_frame * V_frame_i)

    - V_final: The final, calculated Zonos vector.
    - V_utterance: The base vector from the utterance-level direction.
    - V_frame_i: The modifier vector for each frame-level direction.
    - w: User-defined weights to control influence.

    **Methodology:**
    1.  **Set Weights:**
        - w_utterance: Typically 1.0 to give the base emotion its full influence.
        - w_frame: A smaller value (e.g., 0.25) to ensure frame-level events
          act as nuanced modifiers rather than overpowering the base emotion.
    2.  **Calculate Sum:** The weighted utterance vector and all weighted frame
        vectors are summed component-wise.
    3.  **Normalize (Clamp):** The final values are clamped to a valid range
        (e.g., [-1.0, 1.0]) to ensure they are compatible with the TTS model.

    Args:
        llm_output (dict): The JSON output from the LLM for a single utterance,
                           containing 'utterance_level_direction' and 'frame_level_directions'.
        utterance_df (pd.DataFrame): DataFrame containing utterance rules.
        frame_df (pd.DataFrame): DataFrame containing frame rules.
        w_utterance (float): The weight for the utterance-level vector.
        w_frame (float): The weight for each frame-level modifier vector.

    Returns:
        dict: The calculated and clamped final Zonos vector.
    """
    try:
        # Use defaultdict to easily handle addition of new emotion keys
        final_vector = defaultdict(float)
        
        # --- Step 1: Apply the Utterance Level Vector ---
        utterance_key = llm_output['utterance_level_direction']
        
        # Find the corresponding rule in the utterance dataframe
        utterance_rule = utterance_df[utterance_df['stage_direction_key'] == utterance_key]
        if not utterance_rule.empty:
            base_vector_str = utterance_rule.iloc[0]['primary_zonos_vector_value']
            base_vector = json.loads(base_vector_str)
            
            for emotion, value in base_vector.items():
                final_vector[emotion] += value * w_utterance
        else:
            print(f"Warning: Utterance key '{utterance_key}' not found in rules.")

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

# --- 5. Master Prompt Template (with Dynamic Rule Generation) ---
MASTER_PROMPT_TEMPLATE = """You are a master Vocal Director simulating the 'emotion2vec' framework. Your task is to perform a hierarchical analysis of a therapy dialogue.

You must follow all instructions precisely and output a single, valid JSON object inside a JSON code block.

**INSTRUCTIONS:**
1.  Break down the client's speech into a list of separate utterances.
2.  For each utterance, perform a two-level analysis:
    a. **Utterance-Level:** Assign ONE overall stage direction from the "Utterance-Level Vocabulary".
    b. **Frame-Level:** Identify specific words or phrases and assign one or more tags from the "Frame-Level Vocabulary".
3.  ★★★ **MAINTAIN EMOTIONAL COHERENCE:** The emotional direction of an utterance MUST be a logical progression from the previous one. Abrupt, unrealistic emotional shifts are forbidden.

---
★★★ **RULE GENERATION INSTRUCTIONS** ★★★
- If an existing vocabulary entry is a perfect match, **USE IT**.
- If NO existing vocabulary entry accurately captures the emotion, you are AUTHORIZED to **CREATE A NEW ONE**.
- When creating a new rule, you MUST define it completely. For a new utterance-level rule, you MUST provide the `zonos_vector`, `speaking_rate`, and `pitch_std`. For a new frame-level rule, provide the `zonos_vector_value`.
- The new key you create MUST be descriptive and enclosed in brackets, like `[pensive, trailing-off]`.
---

**EXISTING UTTERANCE-LEVEL VOCABULARY:**
{utterance_rules}

**EXISTING FRAME-LEVEL VOCABULARY:**
{frame_rules}

**PERFECT OUTPUT EXAMPLE (Showing usage of an existing rule and creation of a new one):**
```json
{{
  "directed_utterances": [
    {{
      "utterance_text": "I feel like I let the animals down somehow.",
      "is_new_utterance_rule": false,
      "utterance_level_direction": "[sad, slow]",
      "frame_level_directions": [
        {{
            "phrase": "let the animals down",
            "tag": "[cracked-voice]",
            "is_new_frame_rule": false
        }}
      ]
    }},
    {{
      "utterance_text": "Maybe it was all just... pointless.",
      "is_new_utterance_rule": true,
      "utterance_level_direction": "[dejected, quiet-resignation]",
      "new_utterance_rule_definition": {{
          "primary_zonos_vector_value": {{"Sadness": 0.8, "Neutral": 0.2, "Anger": -0.1}},
          "speaking_rate": 8,
          "pitch_std": 50
      }},
      "frame_level_directions": [
        {{
            "phrase": "pointless",
            "tag": "[whisper]",
            "is_new_frame_rule": false
        }}
      ]
    }}
  ]
}}
"""

# --- 6. Prepare Prompts & Main Processing Loop ---
# (This section is mostly the same, but the formatting call needs to be updated)
print(f"Reading input file and preparing prompts from {INPUT_FILE}...")
# (The file reading code is the same as before...)
source_data_list = []
prompts_list = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            source_data_list.append(data)
            dialogue = data.get("dialogue", "")
            if dialogue:
                # We need to escape the curly braces in the prompt itself now
                prompt = MASTER_PROMPT_TEMPLATE.format(
                    utterance_rules=utterance_rules_str,
                    frame_rules=frame_rules_str,
                    full_dialogue=dialogue
                )
                prompts_list.append(prompt)
            else:
                prompts_list.append("")
except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILE}' not found.")
    exit()

if NUM_SAMPLES_TO_PROCESS is not None:
    source_data_list = source_data_list[:NUM_SAMPLES_TO_PROCESS]
    prompts_list = prompts_list[:NUM_SAMPLES_TO_PROCESS]
    print(f"--- Limiting to the first {NUM_SAMPLES_TO_PROCESS} samples for this run. ---")
print(f"Prepared {len(prompts_list)} prompts to process.")

print(f"Starting processing...")
start_time = time.time()

processed_count = 0
try:
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        processed_count = sum(1 for line in f)
except FileNotFoundError:
    pass
if processed_count > 0:
    print(f"Resuming from record #{processed_count + 1}...")
else:
    print("Starting a new processing job.")

remaining_prompts = prompts_list[processed_count:]
remaining_source_data = source_data_list[processed_count:]

with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
    if not remaining_prompts:
        print("All prompts have already been processed.")
    else:
        for i, prompt in enumerate(tqdm(remaining_prompts)):
            llm_output_data = None
            for attempt in range(MAX_RETRIES):
                try:
                    if not prompt:
                        llm_output_data = {}
                        break
                    outputs = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS)
                    generated_text = outputs[0]['generated_text'][len(prompt):].strip()
                    
                    # ★★★★★ NEW, MORE ROBUST PARSING LOGIC ★★★★★
                    # Look for the JSON within the ```json ... ``` block
                    json_block_start = generated_text.find('```json')
                    if json_block_start != -1:
                        # Find the actual start of the JSON object after the fence
                        start_brace = generated_text.find('{', json_block_start)
                        # Find the end of the code block
                        json_block_end = generated_text.find('```', start_brace)
                        if start_brace != -1 and json_block_end != -1:
                            json_string = generated_text[start_brace : json_block_end].strip()
                            llm_output_data = json.loads(json_string)
                            break # Success! Exit retry loop.
                        else:
                            raise ValueError("Found JSON code block but could not extract content.")
                    else:
                        # Fallback for safety, but the model should use the fence
                        start_brace = generated_text.find('{')
                        end_brace = generated_text.rfind('}')
                        if start_brace != -1 and end_brace > start_brace:
                            json_string = generated_text[start_brace : end_brace + 1]
                            llm_output_data = json.loads(json_string)
                            break # Success!
                        else:
                            raise ValueError("Could not find a valid JSON block.")
                
                except Exception as e:
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for item {processed_count + i}. Error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        time.sleep(1)
                    else:
                        print("Max retries reached. Saving placeholder.")
                        llm_output_data = {"directed_utterances": [{"utterance_text": f"Error: Failed after {MAX_RETRIES} retries. Raw output: {generated_text}"}]}

            # --- LUNA'S CALCULATION LOGIC START ---
            # After successfully getting the LLM output, enrich it with our calculations.
            if llm_output_data and 'directed_utterances' in llm_output_data:
                for utterance in llm_output_data['directed_utterances']:
                    # 1. Calculate the final Zonos vector
                    final_vector = calculate_final_vector(utterance, utterance_df, frame_df)
                    utterance['final_zonos_vector'] = final_vector
                    
                    # 2. Add the generation parameters
                    utterance_key = utterance.get('utterance_level_direction')
                    if utterance_key:
                        rule = utterance_df[utterance_df['stage_direction_key'] == utterance_key]
                        if not rule.empty:
                            utterance['generation_parameters'] = {
                                "speaking_rate": float(rule.iloc[0]['speaking_rate']),
                                "pitch_std": float(rule.iloc[0]['pitch_std'])
                            }
            # --- LUNA'S CALCULATION LOGIC END ---

            original_data = remaining_source_data[i]
            result_data = {
                "original_dialogue": original_data,
                "llm_output": llm_output_data
            }
            outfile.write(json.dumps(result_data, ensure_ascii=False) + '\n')

end_time = time.time()
total_records = processed_count + len(remaining_prompts)
print("\n--- Processing Complete! ---")
print(f"Total records in '{OUTPUT_FILE}': {total_records}")
print(f"Processing time for this session: {end_time - start_time:.2f} seconds")