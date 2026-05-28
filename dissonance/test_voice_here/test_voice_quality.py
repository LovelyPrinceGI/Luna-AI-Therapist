"""
Test Zonos voice quality after optimization.
MAX_RETRIES=2, MIN_SECONDS_PER_CHAR=0.025
Tests 15 varied vocal parameter combinations (2-3 sentence utterances).
HOW TO RUN: python dissonance/test_voice_here/test_voice_quality.py
"""
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = r"C:\Luna-AI-Therapist"
sys.path.insert(0, os.path.join(BASE_DIR, "dissonance", "own_script", "dialogue_6"))
from run_synthesis_dialogue_6_module import synth_single_utterance

TEST_DIR = Path(__file__).parent
TMP_JSON = TEST_DIR / "tmp_test_zonos.json"
OUTPUT_VOICE = TEST_DIR / "voice"
OUTPUT_VOICE.mkdir(exist_ok=True)

# =====================================================
# 15 test cases: varied emotions, rates, pitch levels
# Each uses 2-3 sentences matching new client prompt limit
# =====================================================

test_cases = [
    # (name, text, happiness, sadness, fear, anger, speaking_rate, pitch_std)
    ("01_neutral_calm",
     "I've been doing okay lately. Nothing too exciting or terrible going on. Just taking things day by day.",
     0.3, 0.0, 0.0, 0.0, 15, 60),
    
    ("02_sad_depressed",
     "I just feel so empty inside. Nothing really matters anymore. It's hard to even get out of bed.",
     0.0, 0.8, 0.1, 0.0, 10, 25),
    
    ("03_angry_intense",
     "I'm so frustrated right now! Why does this keep happening to me? It's not fair at all.",
     0.0, 0.0, 0.2, 0.8, 22, 140),
    
    ("04_anxious_fearful",
     "I keep worrying about what might go wrong tomorrow. My heart races just thinking about it.",
     0.0, 0.1, 0.7, 0.3, 18, 100),
    
    ("05_happy_upbeat",
     "Things are going really well lately! I got the promotion and my family is healthy.",
     0.7, 0.0, 0.0, 0.0, 16, 80),
    
    ("06_very_slow_depressed",
     "I don't even have the energy to talk. Everything feels so heavy and pointless now.",
     0.0, 0.6, 0.1, 0.0, 10, 20),
    
    ("07_very_fast_angry",
     "I'm absolutely furious right now! This is completely unacceptable! I won't stand for it!",
     0.0, 0.0, 0.3, 0.7, 25, 150),
    
    ("08_surprised_shocked",
     "I can't believe what just happened. This is completely unexpected. I'm in shock.",
     0.0, 0.0, 0.4, 0.1, 20, 130),
    
    ("09_calm_content",
     "I feel peaceful today. The weather is nice and I have nothing urgent to worry about.",
     0.5, 0.0, 0.0, 0.0, 12, 40),
    
    ("10_nervous_anxious",
     "I'm starting to feel really on edge. My stomach is in knots about the meeting tomorrow.",
     0.0, 0.1, 0.6, 0.2, 17, 90),
    
    ("11_defensive_annoyed",
     "I don't see why I have to explain myself. You wouldn't understand what I'm going through.",
     0.0, 0.1, 0.1, 0.5, 19, 70),
    
    ("12_hopeless_despair",
     "I feel completely stuck. Like there's no way out and nothing will ever improve for me.",
     0.0, 0.7, 0.2, 0.0, 11, 30),
    
    ("13_excited_delighted",
     "This is so exciting! I've been waiting for this moment for years. I can't stop smiling!",
     0.8, 0.0, 0.0, 0.0, 20, 110),
    
    ("14_flat_monotone",
     "Things are fine I guess. Work is work. Not much to say about it really.",
     0.0, 0.1, 0.1, 0.0, 13, 25),
    
    ("15_mixed_conflicted",
     "I'm happy about the opportunity but also terrified of failing. I don't know what to feel.",
     0.2, 0.3, 0.3, 0.2, 15, 80),
]

emotion_keys = ["Happiness", "Sadness", "Disgust", "Fear", "Surprise", "Anger", "Other", "Neutral"]

print("=" * 70)
print("VOICE QUALITY TEST - Post-Optimization (MAX_RETRIES=2, MIN_SEC_PER_CHAR=0.025)")
print("=" * 70)
print(f"Output dir: {OUTPUT_VOICE}")
print()

results = []
start_time = time.time()

for idx, (name, text, happy, sad, fear, anger, spk_rate, pitch_std) in enumerate(test_cases):
    print(f"[{idx+1:02d}/15] {name}")
    print(f"  Text: \"{text[:80]}...\"")
    print(f"  Params: happiness={happy}, sadness={sad}, fear={fear}, anger={anger}, rate={spk_rate}, pitch={pitch_std}")
    
    # Build directed_utterance JSON
    emotion_vec = {k: 0.0 for k in emotion_keys}
    emotion_vec["Happiness"] = happy
    emotion_vec["Sadness"] = sad
    emotion_vec["Fear"] = fear
    emotion_vec["Anger"] = anger
    
    directed = {
        "directed_utterances": [{
            "utterance_text": text,
            "is_new_utterance_rule": True,
            "utterance_level_direction": f"[{name}]",
            "new_utterance_rule_definition": {
                "primary_zonos_vector_value": emotion_vec,
                "speaking_rate": spk_rate,
                "pitch_std": pitch_std,
            },
            "frame_level_directions": []
        }]
    }
    
    with open(TMP_JSON, "w", encoding="utf-8") as f:
        json.dump(directed, f, ensure_ascii=False, indent=2)
    
    # Synthesize
    t0 = time.time()
    out_path = synth_single_utterance(1, str(TMP_JSON), dialogue_id=0, prefix="test")
    elapsed = time.time() - t0
    
    if out_path and Path(out_path).exists():
        # Measure actual output
        import librosa
        y, sr = librosa.load(out_path, sr=None)
        dur = len(y) / sr
        rms = float((y ** 2).mean() ** 0.5)
        status = "OK" if rms >= 0.02 and dur >= 0.5 else "LOW"
        results.append({
            "name": name, "dur": dur, "rms": rms,
            "elapsed": elapsed, "status": status
        })
        print(f"  -> {status}: dur={dur:.2f}s, rms={rms:.3f}, synth_time={elapsed:.0f}s")
    else:
        results.append({
            "name": name, "dur": 0, "rms": 0,
            "elapsed": elapsed, "status": "FAIL"
        })
        print(f"  -> FAIL (no audio output)")
    print()

total_time = time.time() - start_time

# Clean up temp JSON
if TMP_JSON.exists():
    TMP_JSON.unlink()

# =====================================================
# Summary
# =====================================================
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'#':>3} {'Name':<25} {'Dur':>6} {'RMS':>6} {'Time':>5}  {'Status':<6}")
print("-" * 65)
pass_count = 0
for i, r in enumerate(results):
    marker = "<- LOW" if r["status"] == "LOW" else ("<- FAIL" if r["status"] == "FAIL" else "")
    if r["status"] == "OK":
        pass_count += 1
    print(f"{i+1:>3} {r['name']:<25} {r['dur']:>5.1f}s {r['rms']:>5.3f} {r['elapsed']:>4.0f}s  {r['status']:<6} {marker}")

print(f"\nPass: {pass_count}/15 | Low: {len([r for r in results if r['status']=='LOW'])} | Fail: {len([r for r in results if r['status']=='FAIL'])}")
print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
print(f"Output files: {list(OUTPUT_VOICE.glob('*.wav')).__len__()} WAVs in {OUTPUT_VOICE}")
