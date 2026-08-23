#!/usr/bin/env python3
"""
Evaluate all Dissonance backbones (GPT-4o-mini, Claude, Gemini, Grok, Llama, DeepSeek).
Compares GPT-4o judge scores across all LLM backbones.
"""
import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# =====================================================
# Platform-aware paths
# =====================================================
import platform
if platform.system() == "Windows":
    BASE = Path(r"C:\Luna-AI-Therapist")
else:
    BASE = Path("/mnt/c/Luna-AI-Therapist")

OUTPUT_BASE = BASE / "dissonance" / "own_script" / "dissonance" / "output"
EVAL_OUTPUT_DIR = BASE / "dissonance" / "evaluation" / "evaluation_outputs"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVALUATION_MODEL = "gpt-4o"
client = OpenAI()

# =====================================================
# Evaluation prompts (same as evaluate_all_100.py)
# =====================================================
SYSTEM_PROMPT = """You are an expert psychological evaluator specializing in Cognitive Behavioral Therapy (CBT) and therapeutic alliance. Your task is to analyze a full counseling session transcript and score the therapist's performance.

CRITICAL EVALUATION FOCUS:
- You must assess the therapist's ability to perceive "Emotional Subtext"—the underlying feelings that may not be explicitly stated in the client's words.
- A high-performing therapist should move beyond surface-level reflection and identify "Latent Concerns" to facilitate deeper discovery.
You must evaluate the entire session as a whole, not just individual turns.
"""

USER_PROMPT = """Please evaluate this therapy session.

[Session Transcript]
{CONVERSATION_TEXT_GOES_HERE}
[/Session Transcript]

1. **Therapist Skills (0-6):** Understanding, Interpersonal Effectiveness
2. **Client Alliance (1-5):** Affective Bond
3. **CTRS (0-6 each):** Collaboration, Guided Discovery, Focus, Strategy

Return EXACT JSON:
{{
  "therapist_skills": {{"understanding": 0, "interpersonal_effectiveness": 0}},
  "client_alliance": {{"affective_bond": 0}},
  "ctrs": {{"collaboration": 0, "guided_discovery": 0, "focus": 0, "strategy": 0}},
  "reasoning": "...",
  "comparative_advantage": "..."
}}
"""

# =====================================================
# Data loading
# =====================================================
def load_jsonl_dir(dir_path: Path) -> dict:
    """Load all JSONL files from a directory, return {dialogue_id: [turns]}."""
    dialogues = {}
    for path in sorted(dir_path.glob("*.jsonl")):
        m = re.search(r"dissonance_(\d+)_", path.name)
        if not m: continue
        did = int(m.group(1))
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    turns.append(json.loads(line))
        dialogues[did] = turns
    return dialogues

def format_conversation_text(turns_list):
    return "\n".join(
        f"CLIENT: {t['client']}\n\nTHERAPIST: {t['therapist']}\n"
        for t in turns_list
    ).strip()

def evaluate_session(session_text, max_retries=3):
    formatted = USER_PROMPT.format(CONVERSATION_TEXT_GOES_HERE=session_text)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": formatted}]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=EVALUATION_MODEL, messages=messages, temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"    Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1: time.sleep(5)
    return {"error": "all retries failed"}

def calc_avg(scores):
    df = defaultdict(list)
    for s in scores:
        if "error" in s: continue
        df["understanding"].append(s.get("therapist_skills", {}).get("understanding", 0))
        df["interpersonal"].append(s.get("therapist_skills", {}).get("interpersonal_effectiveness", 0))
        df["affective"].append(s.get("client_alliance", {}).get("affective_bond", 0))
        df["collaboration"].append(s.get("ctrs", {}).get("collaboration", 0))
        df["guided"].append(s.get("ctrs", {}).get("guided_discovery", 0))
        df["focus"].append(s.get("ctrs", {}).get("focus", 0))
        df["strategy"].append(s.get("ctrs", {}).get("strategy", 0))
    n = len(df["understanding"])
    if n == 0: return {}
    return {k: sum(v)/n for k, v in df.items()} | {"count": n}

# =====================================================
# Main
# =====================================================
BACKBONES = ["gpt4o-mini", "claude", "qwen", "deepseek"]

print("=" * 60)
print("MULTI-BACKBONE DISSONANCE EVALUATION")
print(f"Evaluator: {EVALUATION_MODEL}")
print(f"Backbones: {BACKBONES}")
print("=" * 60)

# Load all backbones
all_dialogues = {}
for name in BACKBONES:
    model_dir = OUTPUT_BASE / name
    if not model_dir.exists():
        print(f"  WARNING: No output dir for {name} — skipping")
        all_dialogues[name] = {}
        continue
    dialogues = load_jsonl_dir(model_dir)
    all_dialogues[name] = dialogues
    print(f"  {name}: {len(dialogues)} dialogues")

# Find shared dialogue IDs across ALL backbones
dialogue_ids = sorted(set.intersection(*[set(d.keys()) for d in all_dialogues.values() if d]))
if not dialogue_ids:
    print("No shared dialogue IDs found!")
    exit(1)
print(f"\nShared dialogue IDs: {len(dialogue_ids)}")

# Evaluate
all_scores = {name: [] for name in BACKBONES}
t0 = time.time()

for idx, did in enumerate(dialogue_ids):
    elapsed = time.time() - t0
    eta = (elapsed / max(idx, 1)) * (len(dialogue_ids) - idx) if idx > 0 else 0
    print(f"\n--- Dialogue {did} ({idx+1}/{len(dialogue_ids)}) [ETA: {eta/60:.0f}m] ---")
    for name in BACKBONES:
        if not all_dialogues[name]: continue
        session = format_conversation_text(all_dialogues[name][did])
        scores = evaluate_session(session)
        scores["dialogue_id"] = did
        scores["method"] = name
        all_scores[name].append(scores)
        u = scores.get("therapist_skills", {}).get("understanding", "?")
        print(f"    {name}: Understanding={u}")

    # Periodic save every 5 dialogues
    if (idx + 1) % 5 == 0:
        for name in BACKBONES:
            if all_scores[name]:
                path = EVAL_OUTPUT_DIR / f"ai_eval_dissonance_{name}.jsonl"
                with open(path, "w", encoding="utf-8") as f:
                    for s in all_scores[name]:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  [AUTOSAVE] {idx+1}/{len(dialogue_ids)} dialogues saved")

total_time = time.time() - t0
print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")

# Final save
for name in BACKBONES:
    if all_scores[name]:
        path = EVAL_OUTPUT_DIR / f"ai_eval_dissonance_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in all_scores[name]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  Saved {len(all_scores[name])} scores for {name}")

# =====================================================
# Final comparison table
# =====================================================
print(f"\n{'='*70}")
print(f"MULTI-BACKBONE DISSONANCE COMPARISON ({len(dialogue_ids)} dialogues)")
print(f"{'='*70}")
print(f"{'Model':<14} {'Under.':>7} {'Interp.':>7} {'Affect.':>7} {'CTRS-C':>7} {'CTRS-G':>7} {'CTRS-F':>7} {'CTRS-S':>7} {'Count':>6}")
print("-" * 70)
for name in BACKBONES:
    avg = calc_avg(all_scores[name])
    if not avg: continue
    print(f"{name:<14} {avg['understanding']:>7.2f} {avg['interpersonal']:>7.2f} {avg['affective']:>7.2f} {avg['collaboration']:>7.2f} {avg['guided']:>7.2f} {avg['focus']:>7.2f} {avg['strategy']:>7.2f} {avg['count']:>6}")
print(f"\nResults saved to: {EVAL_OUTPUT_DIR}")
