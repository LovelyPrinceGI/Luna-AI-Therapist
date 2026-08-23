# Step by step guide to run

# **Set up**

### open wsl ubuntu command:
```python
wsl -d ubuntu-24.04 -u luna
```

### move to dissonance directory:
```python
cd /mnt/c/Luna-AI-Therapist/dissonance
```

### Activate venu
```python
source ~/luna_env/bin/activate
```


# **Implementation**

Run commands to set api key:
```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
```

## Run ALL 4 conditions per backbone (serial):

### Sonnet 4.6
`baseline & emotion`
```bash
python run_multibackbone_baseline.py  --model claude --start 1 --end 100 &
python run_multibackbone_emotion.py   --model claude --start 1 --end 100 &
```
`multimodal`
```bash
python run_multibackbone_multimodal.py --model claude --start 1 --end 100
```
`dissonance`
```bash
python run_dissonance_multibackbone.py --model claude --start 1 --end 100
#--model qwen --model deepseek
```

### qwen
`baseline & emotion`
```bash
python run_multibackbone_baseline.py  --model qwen --start 1 --end 100 &
python run_multibackbone_emotion.py   --model qwen --start 1 --end 100 &
```
`multimodal`
```bash
python run_multibackbone_multimodal.py --model qwen --start 1 --end 100
```
`dissonance`
```bash
python run_dissonance_multibackbone.py --model qwen --start 1 --end 100
```

### DeepSeek
`baseline & emotion`
```bash
python run_multibackbone_baseline.py  --model deepseek --start 1 --end 100 &
python run_multibackbone_emotion.py   --model deepseek --start 1 --end 100 &
```
`multimodal`
```bash
python run_multibackbone_multimodal.py --model deepseek --start 1 --end 100
```
`dissonance`
```bash
python run_dissonance_multibackbone.py --model deepseek --start 1 --end 100
```

## If you want to measure inference timing:
- Add inference_timing script in advance of original command

```bash
python inference_timing.py --model qwen --condition baseline    --start 1 --end 1
python inference_timing.py --model qwen --condition emotion     --start 1 --end 1
python inference_timing.py --model qwen --condition multimodal  --start 1 --end 1
python inference_timing.py --model qwen --condition dissonance  --start 1 --end 1
```

# Before evaluate:
- Check how many successed dialogue example:
```bash
ls -1 /mnt/c/Luna-AI-Therapist/dissonance/multibackbone/llama/dissonance/*.jsonl | wc -l
```

# Evaluate:

### Run in WSL2:
```python
cd /mnt/c/Luna-AI-Therapist/dissonance
```
```python
export OPENAI_API_KEY="sk-..."
```

### Test 2 dialogues first
```python
python evaluation/evaluate_multibackbone_final.py --start 1 --end 2
```

### Full run
```python
python evaluation/evaluate_multibackbone_final.py --start 1 --end 100
```

### Specific backbones only
```python
python evaluation/evaluate_multibackbone_final.py --backbones claude qwen --start 1 --end 10
```

### Output:
```python
evaluation_outputs/
  ai_eval_multibackbone_gpt4o-mini_baseline.jsonl
  ai_eval_multibackbone_gpt4o-mini_emotion.jsonl
  ... (16 files total)
```

