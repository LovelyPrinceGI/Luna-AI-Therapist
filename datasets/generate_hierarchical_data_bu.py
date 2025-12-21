import torch
from transformers import pipeline
import json
import time
from tqdm import tqdm
import pandas as pd
import re

# --- 1. Configuration ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
UTTERANCE_RULES_FILE = "utterance_level_rules.csv"
FRAME_RULES_FILE = "frame_level_rules.csv"
INPUT_FILE = "cactus_huggingface_train.jsonl"
OUTPUT_FILE = "final_hierarchical_output.jsonl" 
MAX_NEW_TOKENS = 2048
NUM_SAMPLES_TO_PROCESS = 2 # Set to None for the full run
MAX_RETRIES = 3

# --- 2. Load Model ---
print(f"Loading model: {MODEL_ID}...")
pipe = pipeline("text-generation", model=MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
print("Model loaded successfully!")

# --- 3. Load and Format Rules from CSV Files ---
print("Loading rules from CSV files...")
try:
    utterance_df = pd.read_csv(UTTERANCE_RULES_FILE)
    utterance_rules_list = [f"- `{row['stage_direction_key']}`: {row['primary_zonos_vector_value']}" for index, row in utterance_df.iterrows()]
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
except FileNotFoundError as e:
    print(f"Error: Could not find rule file: {e}. Please ensure CSV files are in the same directory.")
    exit()

# --- 4. Master Prompt Template ---
# REMOVE the .format() from this definition block
MASTER_PROMPT_TEMPLATE = """You are a master Vocal Director simulating the 'emotion2vec' framework. Your task is to perform a hierarchical analysis of a therapy dialogue.

You must follow these steps precisely:
1. Break down the client's speech into a list of separate utterances.
2. For each utterance, perform a two-level analysis:
    a. **Utterance-Level:** Assign ONE overall stage direction from the "Utterance-Level Vocabulary".
    b. **Frame-Level:** Identify specific words or phrases and assign one or more tags from the "Frame-Level Vocabulary".
3. Calculate the `final_zonos_vector` for the utterance by blending the base vector from the utterance-level tag with the effects from any frame-level tags.
4. You MUST output your final work as a single, valid JSON object, precisely matching the structure in the provided example.

**UTTERANCE-LEVEL VOCABULARY (Choose ONE per utterance):**
{utterance_rules}

**FRAME-LEVEL VOCABULARY (Choose zero or more per utterance):**
{frame_rules}

**PERFECT OUTPUT EXAMPLE:**
{{"directed_utterances": [
    {{"utterance_text": "I feel like I let the animals down somehow.", "utterance_level_direction": "[sad, slow]", "frame_level_directions": [{{"phrase": "let the animals down", "tag": "[cracked-voice]"}}, {{"phrase": "somehow", "tag": "[whisper]"}}], "final_zonos_vector": {{"Sadness": 1.0, "Fear": 0.3, "Neutral": -0.2}} }}
]}}

**TASK:**
- **Full Dialogue:**
"{full_dialogue}"

**OUTPUT (JSON Object Only):**
"""

# --- 5. Prepare Prompts & Main Processing Loop ---
print(f"Reading input file and preparing prompts from {INPUT_FILE}...")
source_data_list = []
prompts_list = []
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            source_data_list.append(data)
            dialogue = data.get("dialogue", "")
            if dialogue:
                # ★★★★★ THIS IS THE CORRECTED LINE ★★★★★
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
                    start_brace = generated_text.find('{')
                    end_brace = generated_text.rfind('}')
                    if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                        json_string = generated_text[start_brace : end_brace + 1]
                        llm_output_data = json.loads(json_string)
                        break
                    else:
                        raise ValueError("Could not find a valid JSON block in LLM output.")
                except Exception as e:
                    print(f"\nAttempt {attempt + 1}/{MAX_RETRIES} failed for item {processed_count + i}. Error: {e}")
                    if attempt < MAX_RETRIES - 1:
                        print("Retrying...")
                        time.sleep(1)
                    else:
                        print("Max retries reached. Saving placeholder.")
                        llm_output_data = {"directed_utterances": [{"utterance_text": f"Error: Failed after {MAX_RETRIES} retries."}]}
            
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