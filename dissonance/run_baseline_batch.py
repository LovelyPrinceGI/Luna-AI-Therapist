#!/usr/bin/env python3
"""
Batch run Baseline (Text-Only) therapist generation for N dialogues.
Resist persona + shared seeds + text VA + discrete emotion.
NO audio/speech — API only (~2h for 100 dialogues).
"""
import argparse
import os
import sys
import json
import math
from pathlib import Path
from typing import Tuple

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =====================================================
# Platform-aware paths
# =====================================================
import platform
if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

OUTPUT_DIR = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# OpenAI
# =====================================================
MODEL = "gpt-4o-mini"
client = OpenAI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================================================
# Text VA: vad-bert
# =====================================================
VAD_MODEL_NAME = "RobroKools/vad-bert"
tokenizer = AutoTokenizer.from_pretrained(VAD_MODEL_NAME)
vad_model = AutoModelForSequenceClassification.from_pretrained(VAD_MODEL_NAME).to(device)
vad_model.eval()

def _to_minus1_1(x, xmin=1.0, xmax=5.0): return float(2 * (x - xmin) / (xmax - xmin) - 1.0)

def get_text_VA(text: str) -> Tuple[float, float]:
    enc = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = vad_model(**enc)
    v_raw, a_raw, d_raw = out.logits.cpu().numpy()[0].tolist()
    return _to_minus1_1(v_raw, 1.0, 5.0), _to_minus1_1(a_raw, 1.0, 5.0)

# =====================================================
# Discrete Emotion: Russell (1980) circumplex
# =====================================================
def get_discrete_emotion(val: float, aro: float) -> str:
    distance = math.sqrt(val**2 + aro**2)
    if distance < 0.15:
        return "Neutral"
    angle = math.degrees(math.atan2(aro, val))
    if angle < 0:
        angle += 360
    centers = [(0, "Happy"), (45, "Excited"), (90, "Tense"),
               (135, "Angry"), (180, "Sad"), (225, "Depressed"),
               (270, "Calm"), (315, "Content")]
    closest = min(centers, key=lambda c: min(abs(angle - c[0]), 360 - abs(angle - c[0])))
    return closest[1]

# =====================================================
# LLM helper
# =====================================================
def chat_once(system_prompt: str, user_prompt: str, history: list = None) -> str:
    import time
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7, max_tokens=512,
            )
            content = resp.choices[0].message.content
            if content and content.strip():
                return content.strip()
            attempt += 1
            print(f"  [RETRY] Empty response, attempt {attempt}")
        except Exception as e:
            attempt += 1
            wait = min(2 ** attempt, 60)
            print(f"  [RETRY] {e}, attempt {attempt}, waiting {wait}s")
            time.sleep(wait)
            continue
        time.sleep(1)

# =====================================================
# Client prompts (resist persona)
# =====================================================
CLIENT_SYSTEM = """
You are a CBT therapy client talking to therapist "Luna".

- You struggle with anxiety, guilt, and loneliness in your life.
- You sometimes feel misunderstood or skeptical about therapy.
- When the therapist suggests reframing, advice, or homework,
  you may partially resist, question it, or bring up obstacles
  (e.g., "I don't think that will work for me", "It's hard because ...").
- Speak in a natural, first-person voice.
- Stay emotionally consistent across turns.
- Describe thoughts, feelings, and situations in 2–3 sentences per turn.
- In each full dialogue, you must focus on only ONE life problem scenario.
- Do not mix multiple problem seeds in the same dialogue.
- Once a problem seed is assigned for a dialogue, keep that same core life problem throughout the whole dialogue.
"""

CLIENT_USER_TEMPLATE_FIRST = """
Start the first message to your therapist.

Describe what has been bothering you lately (2–3 sentences).
You may already feel unsure whether therapy can really help.

Important:
- This dialogue has exactly ONE assigned life problem scenario.
- You must only use the following scenario in this whole dialogue.
- Do not introduce a second major life problem.

Assigned life problem scenario:
{problem_seed}
"""

CLIENT_USER_TEMPLATE_NEXT = """
Therapist just said:
"{therapist_text}"

Continue the conversation as the client.
Describe what you think and feel now in 2–3 sentences.
If the therapist gives advice, interpretations, or homework,
you can question it, express doubts, or explain why it feels difficult.

Important:
- Stay within the same assigned life problem scenario for this whole dialogue.
- Do not switch to a new major life problem.
"""

# =====================================================
# Shared seeds
# =====================================================
SHARED_SEEDS_PATH = OUTPUT_DIR / "shared_seeds.json"
if not SHARED_SEEDS_PATH.exists():
    raise FileNotFoundError(f"Run generate_shared_seeds.py first!\nExpected: {SHARED_SEEDS_PATH}")
with open(SHARED_SEEDS_PATH, "r", encoding="utf-8") as f:
    SHARED_SEEDS = json.load(f)
print(f"Loaded {len(SHARED_SEEDS)} shared seeds from {SHARED_SEEDS_PATH}")

def get_problem_seed(dialogue_id): return SHARED_SEEDS[dialogue_id - 1]

# =====================================================
# Therapist prompts (Baseline — text only)
# =====================================================
THERAPIST_SYSTEM = """
You are "Luna", a warm, empathetic CBT therapist.

Your goals:
- Understand the client's thoughts, emotions, and behaviors.
- Validate their feelings without dismissing or catastrophizing.
- Gently use CBT techniques (identify automatic thoughts, examine evidence,
  explore alternative perspectives, plan small experiments).

Rules:
- Reply in 2–3 sentences.
- Do NOT mention any numeric scores, analytics, or models.
- End most responses with an open question that invites reflection.
"""

THERAPIST_USER_TEMPLATE = """
Client just said:
"{client_text}"

Please write your next therapist response to the client.
"""

# =====================================================
# Save helper
# =====================================================
def save_dialogue_json_and_jsonl(turns, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = base_path.with_suffix(".json")
    jsonl_path = base_path.with_suffix(".jsonl")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(turns, f, ensure_ascii=False, indent=2)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in turns:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [SAVE] {jsonl_path}")

# =====================================================
# Main loop
# =====================================================
def run_single_dialogue(dialogue_id: int, max_turns: int = 10):
    seed = get_problem_seed(dialogue_id)
    print(f"\n=== BASELINE dialogue {dialogue_id} ===")

    turns = []
    conversation_history = []

    # Turn 1
    client_text = chat_once(CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_FIRST.format(problem_seed=seed))
    print(f"CLIENT (t=1): {client_text[:80]}...")

    val_t, aro_t = get_text_VA(client_text)
    text_emo = get_discrete_emotion(val_t, aro_t)
    print(f"  text VA=({val_t:.2f},{aro_t:.2f}) | emo={text_emo}")

    therapist_text = chat_once(THERAPIST_SYSTEM, THERAPIST_USER_TEMPLATE.format(client_text=client_text),
                               history=list(conversation_history))
    print(f"THERAPIST (t=1): {therapist_text[:80]}...")
    conversation_history.append({"role": "user", "content": client_text})
    conversation_history.append({"role": "assistant", "content": therapist_text})

    turns.append({
        "turn": 1, "client": client_text, "therapist": therapist_text,
        "condition": "baseline",
        "val_t": val_t, "aro_t": aro_t, "text_emotion": text_emo,
    })

    # Turns 2-10
    for t in range(2, max_turns + 1):
        client_text = chat_once(CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_NEXT.format(therapist_text=therapist_text))
        print(f"CLIENT (t={t}): {client_text[:80]}...")

        val_t, aro_t = get_text_VA(client_text)
        text_emo = get_discrete_emotion(val_t, aro_t)

        therapist_text = chat_once(THERAPIST_SYSTEM, THERAPIST_USER_TEMPLATE.format(client_text=client_text),
                                   history=list(conversation_history))
        print(f"THERAPIST (t={t}): {therapist_text[:80]}...")
        conversation_history.append({"role": "user", "content": client_text})
        conversation_history.append({"role": "assistant", "content": therapist_text})

        turns.append({
            "turn": t, "client": client_text, "therapist": therapist_text,
            "condition": "baseline",
            "val_t": val_t, "aro_t": aro_t, "text_emotion": text_emo,
        })

    save_dialogue_json_and_jsonl(turns, OUTPUT_DIR / f"baseline_{dialogue_id}_full_baseline")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    print(f"Baseline batch: dialogues {args.start} to {args.end}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model: {MODEL} | Device: {device}")
    print()

    for i in range(args.start, args.end + 1):
        try:
            run_single_dialogue(dialogue_id=i, max_turns=args.max_turns)
        except Exception as e:
            print(f"ERROR on dialogue {i}: {e}")
            continue

    print(f"\nDONE. Dialogues {args.start}-{args.end} complete.")
    print(f"Output: {OUTPUT_DIR}")
