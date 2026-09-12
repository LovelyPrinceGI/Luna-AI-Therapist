#!/usr/bin/env python3
"""
Dialogue completeness checker for Luna multibackbone runs.

For each backbone x condition, verifies dialogues 1..end:
  - file exists and parses
  - file is from the expected run era (--v4-era / --since), else STALE
  - has the expected number of turns (default 10)
  - turn numbers are sequential 1..N
  - every turn has non-empty client + therapist text
  - (audio conditions) every turn has val_s/vocal_descriptors/audio_path fields
  - (audio conditions) referenced wav file exists on disk (warning only)

Usage:
    python check_dialogue_completeness.py --backbone gpt4o-mini
    python check_dialogue_completeness.py --backbone all --v4-era
    python check_dialogue_completeness.py --backbone claude qwen deepseek
    python check_dialogue_completeness.py --backbone gpt4o-mini --conditions dissonance multimodal
    python check_dialogue_completeness.py --backbone gpt4o-mini --end 50
    python check_dialogue_completeness.py --backbone gpt4o-mini --turns 10
    python check_dialogue_completeness.py --backbone gpt4o-mini --since "2026-09-01 20:00"

Notes:
    --v4-era   uses built-in per-backbone cutoffs (start of each backbone's v4 run).
               Files older than the cutoff are reported as STALE (pre-v4 leftovers).
    --since    manual cutoff 'YYYY-MM-DD HH:MM', applied to EVERY backbone checked
               (only sensible when checking a single backbone).
"""
import argparse
import datetime
import json
import platform
from pathlib import Path

if platform.system() == "Windows":
    BASE_DIR = Path(r"C:\Luna-AI-Therapist")
else:
    BASE_DIR = Path("/mnt/c/Luna-AI-Therapist")

MB = BASE_DIR / "dissonance" / "multibackbone"

BACKBONES = ["gpt4o-mini", "claude", "qwen", "deepseek"]

# Start of the v4 run for each backbone; files older than this are pre-v4 (v1/v2/v3 era).
V4_ERA = {
    "gpt4o-mini": datetime.datetime(2026, 9, 1, 20, 0),
    "qwen":       datetime.datetime(2026, 9, 4, 0, 0),
    "deepseek":   datetime.datetime(2026, 9, 6, 0, 0),
    "claude":     datetime.datetime(2026, 9, 8, 20, 0),
}

CONDITIONS = {
    "baseline":   {"folder": "baseline",   "pattern": "baseline_{i}_full_baseline.json",              "audio": False},
    "emotion":    {"folder": "emotion",    "pattern": "emotion_{i}_full_emotion_online.json",         "audio": False},
    "multimodal": {"folder": "multimodal", "pattern": "dissonance_{i}_full_multimodal.json",          "audio": True},
    "dissonance": {"folder": "dissonance", "pattern": "dissonance_{i}_full_dissonance_online.json",   "audio": True},
}

REQUIRED_FIELDS = ["turn", "client", "therapist"]
AUDIO_FIELDS = ["val_t", "aro_t", "val_s", "aro_s", "vocal_descriptors", "audio_path"]

parser = argparse.ArgumentParser()
parser.add_argument("--backbone", nargs="+", default=["gpt4o-mini"], choices=BACKBONES + ["all"],
                    help="one or more backbones, or 'all'")
parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS.keys()), choices=list(CONDITIONS.keys()))
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--turns", type=int, default=10, help="expected turns per dialogue")
parser.add_argument("--since", type=str, default=None,
                    help="manual cutoff 'YYYY-MM-DD HH:MM'; files older than this are STALE (applies to every backbone checked)")
parser.add_argument("--v4-era", action="store_true",
                    help="use built-in per-backbone v4 run-start cutoffs; older files are STALE")
args = parser.parse_args()

SINCE = None
if args.since:
    SINCE = datetime.datetime.strptime(args.since, "%Y-%m-%d %H:%M")

backbones = BACKBONES if "all" in args.backbone else args.backbone


def to_local(p: str) -> Path:
    return Path(p.replace("/mnt/c/", "C:/")) if platform.system() == "Windows" else Path(p)


any_problem = False
problem_summary = []

for bb in backbones:
    cutoff = SINCE if SINCE is not None else (V4_ERA.get(bb) if args.v4_era else None)
    print("=" * 78)
    line = f"COMPLETENESS CHECK | backbone={bb} | dialogues 1-{args.end} | turns={args.turns}"
    if cutoff is not None:
        line += f" | cutoff={cutoff:%Y-%m-%d %H:%M}"
    print(line)
    print("=" * 78)

    bb_problem = False
    for cond in args.conditions:
        cfg = CONDITIONS[cond]
        folder = MB / bb / cfg["folder"]
        missing, stale, incomplete = [], [], []
        wav_missing = 0
        for i in range(1, args.end + 1):
            f = folder / cfg["pattern"].format(i=i)
            if not f.exists():
                missing.append(i)
                continue
            if cutoff is not None:
                mt = datetime.datetime.fromtimestamp(f.stat().st_mtime)
                if mt < cutoff:
                    stale.append(i)
                    continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception as e:
                incomplete.append((i, f"unparseable: {e}"))
                continue
            problems = []
            if len(data) != args.turns:
                problems.append(f"{len(data)}/{args.turns} turns")
            nums = [t.get("turn") for t in data]
            if nums != list(range(1, len(data) + 1)):
                problems.append("turn numbers not sequential")
            for t in data:
                for k in REQUIRED_FIELDS + (AUDIO_FIELDS if cfg["audio"] else []):
                    if t.get(k) in (None, ""):
                        problems.append(f"t{t.get('turn', '?')}: empty/missing '{k}'")
                        break
            if cfg["audio"]:
                for t in data:
                    ap = t.get("audio_path")
                    if ap and not to_local(ap).exists():
                        wav_missing += 1
            if problems:
                incomplete.append((i, "; ".join(problems[:4])))
        ok = args.end - len(missing) - len(stale) - len(incomplete)
        status = "OK" if not missing and not stale and not incomplete else "PROBLEM"
        if status == "PROBLEM":
            bb_problem = True
        print(f"\n[{cond}] {status} — complete {ok}/{args.end}")
        if missing:
            print(f"  MISSING ({len(missing)}): {missing}")
        if stale:
            print(f"  STALE pre-v4 files ({len(stale)}): {stale}")
        for i, why in incomplete:
            print(f"  INCOMPLETE d{i}: {why}")
        if wav_missing:
            print(f"  WARNING: {wav_missing} wav files not found on disk (records still valid)")

    if bb_problem:
        any_problem = True
        problem_summary.append(bb)
    print()

print("=" * 78)
if not any_problem:
    print("ALL COMPLETE — ready for evaluation.")
else:
    print(f"Fix needed for: {', '.join(problem_summary)}")
    print("Rerun the MISSING/STALE/INCOMPLETE dialogue IDs, e.g.:")
    for bb in problem_summary:
        print(f"  python <script_v4>.py --model {bb} --start <id> --end <id>")
print("=" * 78)
