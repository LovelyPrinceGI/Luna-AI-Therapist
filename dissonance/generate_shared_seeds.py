#!/usr/bin/env python3
"""
Generate 100 shared problem seeds — ONE TIME use.
Saves to dissonance/own_script/dissonance/output/shared_seeds.json
NEVER regenerate once all 4 conditions use it.
"""
import json, os, sys
from pathlib import Path
from openai import OpenAI

# Platform-aware paths
import platform
if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

OUTPUT_DIR = BASE_DIR / "dissonance" / "own_script" / "dissonance" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "gpt-4o-mini"
client = OpenAI()  # uses env OPENAI_API_KEY

SEED_GENERATOR_SYSTEM = """
You generate one-sentence life problem scenarios for CBT therapy clients.
Each scenario must be specific, varied, and realistic.

Requirements:
- Be specific (name a person, place, event, or timeline)
- Vary across: work stress, relationships, family guilt, health anxiety,
  self-esteem, grief, life transitions, financial stress, academic pressure,
  social anxiety, caregiver burnout, identity crisis, trauma, loneliness
- Use different phrasings and vocabulary every time
- Return ONLY the problem seed text -- no quotes, no labels, no commentary

Example outputs:
"You are worried about an upcoming performance review after your manager gave you critical feedback last month."
"You feel guilty about not visiting your aging parents more often since moving to another city."
"You cannot stop replaying a fight you had with your best friend at their wedding three weeks ago."
"""

seeds = []
for i in range(100):
    seed = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SEED_GENERATOR_SYSTEM},
            {"role": "user", "content": "Generate a new unique therapy scenario."},
        ],
        temperature=0.9,
        max_tokens=100,
    ).choices[0].message.content.strip()
    seeds.append(seed)
    print(f"[{i+1:03d}/100] {seed[:80]}...")

out_path = OUTPUT_DIR / "shared_seeds.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(seeds, f, ensure_ascii=False, indent=2)

print(f"\nSaved 100 seeds to {out_path}")
print("DO NOT regenerate this file.")
