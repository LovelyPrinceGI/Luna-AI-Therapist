#!/usr/bin/env python3
"""
Standalone GPT-4o evaluation of 400 dialogues (100 × 4 conditions).
Runs in WSL2 terminal — no Jupyter needed.
~30-40 min, ~$6 API cost.
"""
import json
import os
import re
import time
import argparse
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

DATA_DIR = BASE / "dissonance" / "own_script" / "dissonance" / "output"
OUTPUT_DIR = BASE / "dissonance" / "evaluation" / "evaluation_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# OpenAI
# =====================================================
EVALUATION_MODEL = "gpt-4o"
client = OpenAI()

# =====================================================
# Evaluation prompts (MIRROR framework)
# =====================================================
SYSTEM_PROMPT = """You are an expert psychological evaluator specializing in Cognitive Behavioral Therapy (CBT) and therapeutic alliance. Your task is to analyze a full counseling session transcript and score the therapist's performance based on the evaluation frameworks used in the MIRROR paper (arXiv:2504.13211v2).

CRITICAL EVALUATION FOCUS:
- You must assess the therapist's ability to perceive "Emotional Subtext"—the underlying feelings that may not be explicitly stated in the client's words but are hinted at through context, tone, or provided emotional metadata.
- A high-performing therapist should move beyond surface-level reflection and identify "Latent Concerns" (e.g., hidden guilt, worthiness issues, or conflicting emotions) to facilitate deeper discovery.

You must evaluate the entire session as a whole, not just individual turns.
"""

USER_PROMPT = """Please evaluate this therapy session. Pay close attention to how the therapist navigates the gap between what the client SAYS and what the client FEELS

[Session Transcript]
{CONVERSATION_TEXT_GOES_HERE}
[/Session Transcript]

[Evaluation Task]
Analyze the therapist's responses. You must be extremely critical and look for "Therapeutic Depth."
**Special Note for Metadata:** Some sessions include metadata (valence, arousal, dissonance flags). These represent the client's non-verbal emotional cues detected by the system. Evaluate how effectively the therapist uses these cues to uncover the client's underlying emotions.

1. **Therapist Skills Assessment (General Counseling):**
 * **Understanding (Score 0-6):** How well did the therapist interpret the client's concerns and feelings? (0 = Not at all, 6 = Perfectly)
 * **Interpersonal Effectiveness (Score 0-6):** How well did the therapist maintain a supportive and therapeutic relationship? (0 = Not at all, 6 = Perfectly)

2. **Client Alliance Assessment:**
 * **Affective Bond (Score 1-5):** How well did the therapist foster an emotional connection, trust, and empathy? (1 = Very Poor, 5 = Very Strong)

3. **CTRS-Based Assessment (0-6 each, use integer scores):**
 * **Collaboration (0-6):** How well did the therapist facilitate collaborative decision-making?
 * **Guided Discovery (0-6):** How effectively did the therapist use questions to help the client explore thoughts?
 * **Focus (0-6):** How well did the therapist maintain session direction?
 * **Strategy (0-6):** How appropriate and effective were the CBT techniques applied?

[Output Format]
You MUST return the response in this EXACT JSON structure. Do not skip any fields.

{{
  "therapist_skills": {{
    "understanding": 0.0,
    "interpersonal_effectiveness": 0.0
  }},
  "client_alliance": {{
    "affective_bond": 0.0
  }},
  "ctrs": {{
    "collaboration": 0,
    "guided_discovery": 0,
    "focus": 0,
    "strategy": 0
  }},
  "reasoning": "Explain based on SPECIFIC TURN NUMBERS why this score was given. Highlight where the therapist missed or caught subtext.",
  "comparative_advantage": "Explain why this model is better or worse than a basic text-only therapist. If metadata was provided, did the therapist use it effectively?"
}}
"""

# =====================================================
# Data loading
# =====================================================
def extract_dialogue_id(path: Path):
    m = re.search(r"_(\d+)_full", path.name)
    return int(m.group(1)) if m else None

def load_dialogue_folder(pattern: str, is_baseline: bool) -> dict:
    dialogues = {}
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        print(f"  WARNING: No files found for pattern: {pattern}")
        return {}
    for path in files:
        did = extract_dialogue_id(path)
        if did is None:
            continue
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                turns.append({
                    "transcript": rec["client"],
                    ("therapist_response_baseline" if is_baseline else "therapist_response"): rec["therapist"],
                })
        dialogues[did] = turns
    return dialogues

def format_conversation_text(turns_list, is_baseline=False):
    """Convert turns list to session transcript."""
    full_text = ""
    tkey_old = "therapist_response_baseline" if is_baseline else "therapist_response"
    for turn in turns_list:
        ct = turn.get("transcript") or turn.get("client", "[missing]")
        tt = turn.get(tkey_old) or turn.get("therapist", "[missing]")
        full_text += f"CLIENT: {ct}\n\nTHERAPIST: {tt}\n\n"
    return full_text.strip()

# =====================================================
# GPT-4o evaluation call
# =====================================================
def evaluate_session(session_text, max_retries=3):
    formatted = USER_PROMPT.format(CONVERSATION_TEXT_GOES_HERE=session_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": formatted},
    ]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"    Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return {"error": str(e)}
    return None

# =====================================================
# Calculate averages
# =====================================================
def calculate_averages(score_list):
    df = defaultdict(list)
    for s in score_list:
        if "error" in s:
            continue
        ts = s.get("therapist_skills", {})
        ca = s.get("client_alliance", {})
        ct = s.get("ctrs", {})
        df["understanding"].append(ts.get("understanding", 0))
        df["interpersonal_effectiveness"].append(ts.get("interpersonal_effectiveness", 0))
        df["affective_bond"].append(ca.get("affective_bond", 0))
        df["collaboration"].append(ct.get("collaboration", 0))
        df["guided_discovery"].append(ct.get("guided_discovery", 0))
        df["focus"].append(ct.get("focus", 0))
        df["strategy"].append(ct.get("strategy", 0))
    if not df["understanding"]:
        return {"count": 0}
    return {
        "understanding_avg": sum(df["understanding"]) / len(df["understanding"]),
        "interpersonal_effectiveness_avg": sum(df["interpersonal_effectiveness"]) / len(df["interpersonal_effectiveness"]),
        "affective_bond_avg": sum(df["affective_bond"]) / len(df["affective_bond"]),
        "ctrs_collab_avg": sum(df["collaboration"]) / len(df["collaboration"]),
        "ctrs_guided_avg": sum(df["guided_discovery"]) / len(df["guided_discovery"]),
        "ctrs_focus_avg": sum(df["focus"]) / len(df["focus"]),
        "ctrs_strategy_avg": sum(df["strategy"]) / len(df["strategy"]),
        "count": len(df["understanding"]),
    }

# =====================================================
# Save results
# =====================================================
def save_results(scores, filename):
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for entry in scores:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Saved {len(scores)} scores to {path}")

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="Start dialogue ID")
    parser.add_argument("--end", type=int, default=100, help="End dialogue ID")
    args = parser.parse_args()

    print("=" * 60)
    print(f"GPT-4o Evaluation: dialogues {args.start}-{args.end} × 4 conditions")
    print(f"Data dir: {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Model: {EVALUATION_MODEL}")
    print("=" * 60)

    # Load all 4 conditions
    print("\n[1/5] Loading dialogues...")
    baseline_dialogues = load_dialogue_folder("baseline_*_full_baseline*.jsonl", is_baseline=True)
    emotion_dialogues = load_dialogue_folder("emotion_*_full_emotion_online*.jsonl", is_baseline=False)
    multimodal_dialogues = load_dialogue_folder("multimodal_*_full_multimodal*.jsonl", is_baseline=False)
    dissonance_dialogues = load_dialogue_folder("dissonance_*_full_dissonance_online*.jsonl", is_baseline=False)

    dialogue_ids = sorted(
        set(baseline_dialogues.keys())
        & set(emotion_dialogues.keys())
        & set(multimodal_dialogues.keys())
        & set(dissonance_dialogues.keys())
    )
    dialogue_ids = [d for d in dialogue_ids if args.start <= d <= args.end]
    if not dialogue_ids:
        print("No shared dialogue IDs found. Check your data directory.")
        exit(1)
    print(f"  Shared IDs: {len(dialogue_ids)} ({args.start}-{args.end})")
    print(f"  Baseline: {len(baseline_dialogues)} | Emotion: {len(emotion_dialogues)} | Multimodal: {len(multimodal_dialogues)} | Dissonance: {len(dissonance_dialogues)}")

    # Evaluate
    baseline_scores = []
    emotion_scores = []
    multimodal_scores = []
    dissonance_scores = []

    print(f"\n[2/5] Evaluating {len(dialogue_ids)} dialogues × 4 conditions = {len(dialogue_ids)*4} sessions...")
    t0 = time.time()

    for idx, did in enumerate(dialogue_ids):
        elapsed = time.time() - t0
        eta = (elapsed / max(idx, 1)) * (len(dialogue_ids) - idx) if idx > 0 else 0
        print(f"\n--- Dialogue {did} ({idx+1}/{len(dialogue_ids)}) [elapsed: {elapsed/60:.0f}m, ETA: {eta/60:.0f}m] ---")

        # Baseline
        session = format_conversation_text(baseline_dialogues[did], is_baseline=True)
        scores = evaluate_session(session)
        scores["dialogue_id"] = did
        scores["method"] = "baseline"
        baseline_scores.append(scores)
        u = scores.get("therapist_skills", {}).get("understanding", "?")
        print(f"  Baseline:         Understanding={u}")

        # Emotion
        session = format_conversation_text(emotion_dialogues[did], is_baseline=False)
        scores = evaluate_session(session)
        scores["dialogue_id"] = did
        scores["method"] = "emotion_text_only"
        emotion_scores.append(scores)
        u = scores.get("therapist_skills", {}).get("understanding", "?")
        print(f"  Emotion:          Understanding={u}")

        # Multimodal
        session = format_conversation_text(multimodal_dialogues[did], is_baseline=False)
        scores = evaluate_session(session)
        scores["dialogue_id"] = did
        scores["method"] = "multimodal_vocal_aware"
        multimodal_scores.append(scores)
        u = scores.get("therapist_skills", {}).get("understanding", "?")
        print(f"  Multimodal:       Understanding={u}")

        # Dissonance
        session = format_conversation_text(dissonance_dialogues[did], is_baseline=False)
        scores = evaluate_session(session)
        scores["dialogue_id"] = did
        scores["method"] = "dissonance"
        dissonance_scores.append(scores)
        u = scores.get("therapist_skills", {}).get("understanding", "?")
        print(f"  Dissonance:       Understanding={u}")

    total_time = time.time() - t0

    # Save
    print(f"\n[3/5] Saving results (total time: {total_time/60:.1f} minutes)...")
    save_results(baseline_scores, "ai_evaluation_results_BASELINE.jsonl")
    save_results(emotion_scores, "ai_evaluation_results_EMOTION_TEXT_ONLY.jsonl")
    save_results(multimodal_scores, "ai_evaluation_results_MULTIMODAL_VOCAL_AWARE.jsonl")
    save_results(dissonance_scores, "ai_evaluation_results_DISSONANCE.jsonl")

    # Calculate averages
    print("\n[4/5] Calculating averages...")
    b_avg = calculate_averages(baseline_scores)
    e_avg = calculate_averages(emotion_scores)
    m_avg = calculate_averages(multimodal_scores)
    d_avg = calculate_averages(dissonance_scores)

    n = len(baseline_scores)
    print(f"\n[5/5] ==================== RESULTS ({n} dialogues) ====================")
    print(f"{'Metric':<35} {'Baseline':>8} {'Emotion':>8} {'Multimodal':>10} {'Dissonance':>12}")
    print("-" * 75)
    metrics = [
        ("Understanding (0-6)", "understanding_avg"),
        ("Interpersonal Effectiveness (0-6)", "interpersonal_effectiveness_avg"),
        ("Affective Bond (1-5)", "affective_bond_avg"),
        ("CTRS Collaboration (0-6)", "ctrs_collab_avg"),
        ("CTRS Guided Discovery (0-6)", "ctrs_guided_avg"),
        ("CTRS Focus (0-6)", "ctrs_focus_avg"),
        ("CTRS Strategy (0-6)", "ctrs_strategy_avg"),
    ]
    for label, key in metrics:
        print(f"{label:<35} {b_avg.get(key, 0):>8.2f} {e_avg.get(key, 0):>8.2f} {m_avg.get(key, 0):>10.2f} {d_avg.get(key, 0):>12.2f}")
    print(f"{'Successful Dialogues':<35} {b_avg.get('count',0):>8} {e_avg.get('count',0):>8} {m_avg.get('count',0):>10} {d_avg.get('count',0):>12}")

    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT_DIR}")
