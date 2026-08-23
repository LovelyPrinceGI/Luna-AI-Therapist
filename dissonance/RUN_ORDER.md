# WSL2 Run Order — 4 Condition Batch Pipeline

## 6 Steps to Generate 400 Dialogues

### Step 1: Open WSL2 + Activate Environment

```powershell
wsl -d Ubuntu-24.04 -u luna
```

```bash
source ~/luna_env/bin/activate
cd /mnt/c/Luna-AI-Therapist/dissonance
```

### Step 2: Set OpenAI Key

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Step 3: Generate Shared Seeds (~2 min, ONE TIME)

```bash
python generate_shared_seeds.py
```

Output: `dissonance/own_script/dissonance/output/shared_seeds.json` ← ห้ามทิ้ง ห้าม regenerate

### Step 4: Start Dissonance-Aware (~25h, GPU)

```bash
python run_dissonance_batch.py --start 1 --end 100
```

### Step 5: Open Terminal 2 for Baseline + Emotion (~2h, API only)

```bash
source ~/luna_env/bin/activate
export OPENAI_API_KEY="sk-your-key-here"
cd /mnt/c/Luna-AI-Therapist/dissonance

python run_baseline_batch.py --start 1 --end 100 &
python run_emotion_batch.py --start 1 --end 100 &
```

### Step 6: After Dissonance finishes → Multimodal (~25h, GPU)

```bash
python run_multimodal_batch.py --start 1 --end 100
```
### Step 7: Run evaluation
```bash
python evaluation/evaluate_all_100.py --start 1 --end 100
```

---

## Timeline

| Day | What | Progress |
|-----|------|:--:|
| Day 1 | Seeds + start Dissonance + Baseline/Emotion | — |
| Day 1 afternoon | Baseline + Emotion done | ✅ 200/400 |
| Day 2 morning | Dissonance done | ✅ 300/400 |
| Day 2 morning | Start Multimodal | — |
| Day 3 morning | Multimodal done | ✅ 400/400 |

## Output Structure

```bash
dissonance/own_script/dissonance/output/
├── shared_seeds.json                          ← PERMANENT, committed to git
├── dissonance_{1..100}_full_dissonance_online.json/.jsonl
├── multimodal_{1..100}_full_multimodal.json/.jsonl
├── baseline_{1..100}_full_baseline.json/.jsonl
└── emotion_{1..100}_full_emotion_online.json/.jsonl
```

## Batch Scripts

| File | Type | Zonos/WavLM | Runtime |
|------|:--:|:--:|:--:|
| `generate_shared_seeds.py` | 🆕 Generator | ❌ | 2 min |
| `run_baseline_batch.py` | 🆕 Baseline | ❌ | ~2h |
| `run_emotion_batch.py` | 🆕 Emotion | ❌ | ~2h |
| `run_dissonance_batch.py` | ✏️ Dissonance | ✅ | ~25h |
| `run_multimodal_batch.py` | ✏️ Multimodal | ✅ | ~25h |

## Condition Comparison

| Feature | Baseline | Emotion | Multimodal | Dissonance |
|---------|:--:|:--:|:--:|:--:|
| Seeds | shared | shared | shared | shared |
| Resist persona | ✅ | ✅ | ✅ | ✅ |
| 2-3 sentences (client) | ✅ | ✅ | ✅ | ✅ |
| 2-3 sentences (therapist) | ✅ | ✅ | ✅ | ✅ |
| vad-bert text VA | ✅ | ✅ | ✅ | ✅ |
| Russell discrete emotion | ✅ | ✅ | ✅ | ✅ |
| WavLM speech VA | ❌ | ❌ | ✅ | ✅ |
| Vocal descriptors (librosa) | ❌ | ❌ | ✅ | ✅ |
| Delta / is_dissonant | ❌ | ❌ | ❌ | ✅ |
| Emotion mismatch | ❌ | ❌ | ❌ | ✅ |
| audio_path | ❌ | ❌ | ✅ | ✅ |

## WSL2 Resume Card

```powershell
wsl -d Ubuntu-24.04 -u luna
```

```bash
source ~/luna_env/bin/activate
export OPENAI_API_KEY="sk-..."
cd /mnt/c/Luna-AI-Therapist/dissonance
```
