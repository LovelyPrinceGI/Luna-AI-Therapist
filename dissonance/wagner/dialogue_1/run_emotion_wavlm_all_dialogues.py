import os
import re
import csv

import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification


# รูทโฟลเดอร์ที่มี dialogue_1, dialogue_2, ...
AUDIO_ROOT = r"c:\Users\Legion 5 Pro\OneDrive\Documents\Graduate research\test\own_script"

# ไฟล์ผลลัพธ์รวมทุก dialogue (ตั้งชื่อใหม่กันชน audeering)
OUT_CSV = r"c:\Users\Legion 5 Pro\OneDrive\Documents\Graduate research\test\wagner\emotion_results_wavlm_all.csv"

MODEL_NAME = "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes"

# ชื่อไฟล์เสียง: dialogue_X_utterance_Y.wav
fname_re = re.compile(r"dialogue_(\d+)_utterance_(\d+)\.wav")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Loading model {MODEL_NAME} on {device} ...")
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    target_sr = model.config.sampling_rate
    mean = model.config.mean
    std = model.config.std
    id2label = model.config.id2label  # {0: 'arousal', 1: 'dominance', 2: 'valence'}
    print("id2label:", id2label)
    return model, target_sr, mean, std


def predict_file(path, model, target_sr, mean, std):
    # อ่านด้วย soundfile
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # resample ด้วย librosa ให้ตรง sr โมเดล
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # normalize ตาม usage
    audio = (audio - mean) / (std + 1e-6)

    # เตรียม tensor
    wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    mask = torch.ones(1, wavs.shape[1], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(wavs, mask)    # pred: tensor [1,3]

    logits = pred.cpu().numpy()[0].astype(float)  # [A, D, V] ~ 0..1
    aro = float(logits[0])
    dom = float(logits[1])
    val = float(logits[2])

    # ถ้าอยาก clip 0..1 ด้วย (เลือกใช้ / ไม่ใช้ก็ได้)
    # aro, dom, val = [float(x) for x in np.clip(logits, 0.0, 1.0)]

    return aro, dom, val


def main():
    model, target_sr, mean, std = load_model()
    rows = []

    # วนทุกโฟลเดอร์ dialogue_*
    for dialogue_dir in sorted(os.listdir(AUDIO_ROOT)):
        full_dialogue_path = os.path.join(AUDIO_ROOT, dialogue_dir, "voice")
        if not os.path.isdir(full_dialogue_path):
            continue
        if not dialogue_dir.startswith("dialogue_"):
            continue

        try:
            dialogue_id_int = int(dialogue_dir.split("_")[1])
        except Exception:
            print("Skip folder (cannot parse id):", dialogue_dir)
            continue

        print(f"\n=== Processing {dialogue_dir} (id {dialogue_id_int}) ===")

        for fname in sorted(os.listdir(full_dialogue_path)):
            if not fname.endswith(".wav"):
                continue

            m = fname_re.match(fname)
            if not m:
                print("  Skip (pattern mismatch):", fname)
                continue

            _dialogue_id_from_file, utt_id = m.groups()
            path = os.path.join(full_dialogue_path, fname)

            try:
                aro, dom, val = predict_file(path, model, target_sr, mean, std)
                # ถ้าสนใจแค่ V/A ก็ ignore dom ตรงนี้ได้
                print(f"  {fname}\tA={aro:.3f}\tV={val:.3f}")

                rows.append({
                    "filename": fname,
                    "dialogue_id": int(dialogue_id_int),
                    "utterance_id": int(utt_id),
                    "arousal": aro,
                    "valence": val,
                    "dominance": dom,  # ถ้าไม่อยากเก็บ dominance ลบคีย์นี้ออกได้
                })
            except Exception as e:
                print(f"  Error on {fname}: {e}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["filename", "dialogue_id", "utterance_id", "arousal", "valence", "dominance"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
