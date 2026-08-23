### open wsl ubuntu command:
```python
-d ubuntu-24.04 -u luna
```

### move to dissonance directory:
```python
cd /mnt/c/Luna-AI-Therapist/dissonance
```

### Activate venu
```python
source ~/luna_env/bin/activate
```

### Run Dissonance first (~5 hours)
```python
python run_dissonance_batch.py --start 1 --end 100
```

### Then Multimodal (~5 hours)
```python
python run_multimodal_batch.py --start 1 --end 100
```


# **For new restuructured**

Run commands:
```bash
export OPENROUTER_API_KEY="sk-or-..."
export OPENAI_API_KEY="sk-..."
cd /mnt/c/Luna-AI-Therapist/dissonance
```

## Run ALL 4 conditions per backbone (serial):

### Haiku
`baseline & emotion`
```bash
python run_multibackbone_baseline.py  --model haiku --start 1 --end 100 &
python run_multibackbone_emotion.py   --model haiku --start 1 --end 100 &
```
`multimodal`
```bash
python run_multibackbone_multimodal.py --model haiku --start 1 --end 100
```
`dissonance`
```bash
python run_dissonance_multibackbone.py --model haiku --start 1 --end 100
#--model llama --model deepseek
```

### Llama
`baseline & emotion`
```bash
python run_multibackbone_baseline.py  --model llama --start 1 --end 100 &
python run_multibackbone_emotion.py   --model llama --start 1 --end 100 &
```
`multimoda;`
```bash
python run_multibackbone_multimodal.py --model llama --start 1 --end 100
```
`dissonance`
```bash
python run_dissonance_multibackbone.py --model llama --start 1 --end 100
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
python run_dissonance_multibackbone.py --model deepseek --start 1 --end 10
```

# Evaluate:
```bash
python evaluation/evaluate_multibackbone.py
```