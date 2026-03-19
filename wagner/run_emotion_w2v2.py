import os
import torch
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2Processor
from transformers import AutoModelForAudioClassification as EmotionModel

AUDIO_DIR = "final_audio_output_final"
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model {MODEL_NAME} on {device} ...")
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = EmotionModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

TARGET_SR = 16000

def predict_file(path):
    # อ่านเสียงด้วย soundfile
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # แปลงเป็น tensor แล้ว resample ถ้าจำเป็น
    audio_tensor = torch.tensor(audio, dtype=torch.float32)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        # เพิ่ม batch dimension = 1 แล้วค่อย squeeze กลับ
        audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        sr = TARGET_SR

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
    for fname in sorted(os.listdir(AUDIO_DIR)):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(AUDIO_DIR, fname)
        try:
            arousal, dominance, valence = predict_file(path)
            print(f"{fname}\tA={arousal:.3f}\tD={dominance:.3f}\tV={valence:.3f}")
        except Exception as e:
            print(f"Error on {fname}: {e}")

if __name__ == "__main__":
    main()
