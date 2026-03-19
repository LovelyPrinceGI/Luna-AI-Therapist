import os
import pandas as pd

BASE_DIR = "/home/patsakornt/work/test"
DISSONANCE_DIR = os.path.join(BASE_DIR, "dissonance")
OWN_SCRIPT_DIR = os.path.join(DISSONANCE_DIR, "own_script")
WAGNER_DIR = os.path.join(DISSONANCE_DIR, "wagner")

def load_text_vad(dialogue_id: int) -> pd.DataFrame:
    ddir = os.path.join(OWN_SCRIPT_DIR, f"dialogue_{dialogue_id}")
    path = os.path.join(ddir, f"dialogue_{dialogue_id}_vad_text.csv")
    return pd.read_csv(path)

def load_speech_vad(dialogue_id: int) -> pd.DataFrame:
    ddir = os.path.join(WAGNER_DIR, f"dialogue_{dialogue_id}")
    path = os.path.join(ddir, "emotion_results_wavlm_all.csv")
    return pd.read_csv(path)

def compute_dissonance_for_dialogue(dialogue_id: int) -> pd.DataFrame:
    df_text = load_text_vad(dialogue_id)
    df_speech = load_speech_vad(dialogue_id)

    # ปรับชื่อ column / normalize ให้เหมือนใน compare_dissonance-7.ipynb
    # ตรงนี้คุณ copy logic normalization+scaling จาก notebook มาได้เลย
    # สำคัญ: ต้องให้มีคอลัมน์ utterance_id ทั้งสองฝั่งแล้ว merge กัน

    # ตัวอย่างคร่าว ๆ (ให้แก้ให้ตรงของจริง):
    df_speech = df_speech.rename(columns={"arousal_scaled": "aro_s",
                                          "valence_scaled": "val_s"})
    df_text = df_text.rename(columns={"arousal_scaled": "aro_t",
                                      "valence_scaled": "val_t"})
    merged = pd.merge(df_speech, df_text, on="utterance_id", suffixes=("_speech","_text"))

    merged["delta_arousal"] = (merged["aro_s"] - merged["aro_t"]).abs()
    merged["delta_valence"] = (merged["val_s"] - merged["val_t"]).abs()

    # define threshold ตามที่คุณใช้ใน notebook
    thr_aro = 0.5
    thr_val = 0.5
    merged["dissonant_arousal"] = merged["delta_arousal"] > thr_aro
    merged["dissonant_valence"] = merged["delta_valence"] > thr_val
    merged["dissonant_any"] = merged["dissonant_arousal"] | merged["dissonant_valence"]

    merged["dialogue_id"] = dialogue_id
    return merged[["dialogue_id", "utterance_id",
                   "aro_s","val_s","aro_t","val_t",
                   "delta_arousal","delta_valence",
                   "dissonant_arousal","dissonant_valence","dissonant_any"]]

def save_dissonance(dialogue_id: int, df: pd.DataFrame):
    out_dir = os.path.join(OWN_SCRIPT_DIR, f"dialogue_{dialogue_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"dialogue_{dialogue_id}_dissonance.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved dissonance to {out_path}")

def run_all(dialogue_ids):
    for d in dialogue_ids:
        df = compute_dissonance_for_dialogue(d)
        save_dissonance(d, df)

if __name__ == "__main__":
    # สมมติว่ามี dialogue_1 .. dialogue_20
    run_all(range(1, 21))
