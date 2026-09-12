#!/usr/bin/env python3
"""
Post-run signal check: does the dissonance channel track true masking now?

Reads the NEW gpt4o-mini d1-10 dissonance data and reports:
  1. DSP usage (rate + kinds)
  2. corr(zonos_neg_vector, measured val_s)   -> should be clearly negative
  3. Flag rate                                 -> expect ~20-35%
  4. Precision: flagged turns that are TRUE masking (heuristic ground truth
     from internal_thought: negative hidden words + non-negative text valence)
  5. Recall: true-masking turns that were flagged

Usage (WSL or Windows):
    python check_signal_alignment.py
"""
import json
import math
import platform
from pathlib import Path

if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

BASE = BASE_DIR / "dissonance" / "multibackbone" / "gpt4o-mini" / "dissonance"

NEG_WORDS = ["scared", "afraid", "terrified", "anxious", "anxiety", "overwhelmed",
             "hurt", "sad", "ashamed", "guilty", "lonely", "hate", "cry", "panic",
             "insecure", "worthless", "drowning", "lost", "struggling", "weak", "fail"]

def is_true_masking(turn):
    it = (turn.get("internal_thought") or "").lower()
    has_neg = any(w in it for w in NEG_WORDS)
    return has_neg and turn["val_t"] >= 0.0

def corr(a, b):
    if len(a) < 3:
        return 0.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db else 0.0

tot = 0
dsp_kinds = {}
negv, vals = [], []
flags = 0
true_mask = 0
flag_and_mask = 0
mask_caught = 0
flagged_turns = []

for i in range(1, 11):
    p = BASE / f"dissonance_{i}_full_dissonance_online.json"
    if not p.exists():
        print(f"[WARN] missing {p.name}")
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    for t in data:
        tot += 1
        k = t.get("emotion_dsp")
        if k:
            dsp_kinds[k] = dsp_kinds.get(k, 0) + 1
        zd = (t.get("zonos_director") or {}).get("primary_zonos_vector_value", {})
        if zd:
            negv.append(max(zd.get("Sadness", 0), zd.get("Fear", 0), zd.get("Anger", 0)))
            vals.append(t["val_s"])
        masked = is_true_masking(t)
        flagged = bool(t.get("is_dissonant"))
        if masked:
            true_mask += 1
            if flagged:
                mask_caught += 1
        if flagged:
            flags += 1
            if masked:
                flag_and_mask += 1
            flagged_turns.append((i, t["turn"], "MASK" if masked else "non-mask"))

print("=" * 70)
print("SIGNAL ALIGNMENT CHECK (gpt4o-mini dissonance d1-10)")
print("=" * 70)
print(f"turns analyzed: {tot}")
print(f"DSP applied: {sum(dsp_kinds.values())} ({100*sum(dsp_kinds.values())/max(tot,1):.0f}%) kinds={dsp_kinds}")
print(f"corr(zonos_neg, val_s) = {corr(negv, vals):+.3f}   (target: clearly negative)")
print(f"flag rate = {flags}/{tot} ({100*flags/max(tot,1):.0f}%)   (expect ~20-35%)")
print(f"true masking (heuristic) = {true_mask}/{tot} ({100*true_mask/max(tot,1):.0f}%)")
if flags:
    print(f"PRECISION = {flag_and_mask}/{flags} ({100*flag_and_mask/flags:.0f}%) of flags are true masking")
if true_mask:
    print(f"RECALL    = {mask_caught}/{true_mask} ({100*mask_caught/true_mask:.0f}%) of true masking caught")
print()
print("flagged turns:", flagged_turns)
print()
print("GUIDE:")
print("- corr < -0.25, flag rate 20-35%, precision >= 60%  => signal healthy, evaluate")
print("- flag rate > 50%  => too loose (raise threshold or reduce DSP strength)")
print("- recall < 30%     => still missing most masking (needs stronger DSP)")
