import os
import re
import csv
import torch
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2Processor
from transformers import AutoModelForAudioClassification as EmotionModel

AUDIO_DIR = "final_audio_output_final"
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
OUT_CSV = "wagner/emotion_results_w2v2.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model {MODEL_NAME} on {device} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = EmotionModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

TARGET_SR = 16000
fname_re = re.compile(r"dialogue_(\d+)_utterance_(\d+)\.wav")

def predict_file(path, sr_expected=TARGET_SR):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    if sr != sr_expected:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr_expected)
        audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        sr = sr_expected

    inputs = processor(
        audio_tensor.numpy(),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    logits = outputs.logits.cpu().numpy()[0]
    arousal, dominance, valence = logits.tolist()
    return arousal, dominance, valence

def main():
    rows = []
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if not fname.endswith(".wav"):
            continue

        m = fname_re.match(fname)
        if not m:
            print(f"Skip {fname} (pattern not matched)")
            continue
        dialogue_id, utt_id = m.groups()

        path = os.path.join(AUDIO_DIR, fname)
        try:
            arousal, dominance, valence = predict_file(path)
            print(f"{fname}\tA={arousal:.3f}\tD={dominance:.3f}\tV={valence:.3f}")
            rows.append({
                "filename": fname,
                "dialogue_id": int(dialogue_id),
                "utterance_id": int(utt_id),
                "arousal": arousal,
                "dominance": dominance,
                "valence": valence,
            })
        except Exception as e:
            print(f"Error on {fname}: {e}")

    # เขียน CSV
    fieldnames = ["filename", "dialogue_id", "utterance_id", "arousal", "dominance", "valence"]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
