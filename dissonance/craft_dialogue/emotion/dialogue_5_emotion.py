#!/usr/bin/env python
"""
Run dialogue_5: therapist emotion-aware (online text VA).
Client = LLM, Therapist = LLM ที่เห็น VA จากข้อความของ client แต่ละเทิร์น
"""

import os
import json
import getpass
from typing import Tuple
from openai import OpenAI

MODEL = "gpt-4o-mini"   # เปลี่ยนรุ่นได้ตามใจ

# ==============================
# 1) OpenAI client
# ==============================
def setup_client() -> OpenAI:
    # ล้างค่าเก่าใน env ถ้ามี
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]

    # บังคับถาม key ใหม่ทุกครั้งที่รันสคริปต์
    key = getpass.getpass("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = key

    return OpenAI()


client = setup_client()

# ==============================
# 2) ฟังก์ชันหา VA จาก text (ใช้ RobroKools/vad-bert)
# ==============================
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "RobroKools/vad-bert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ช่วงค่า VAD ดิบของโมเดล (ตาม paper/ตัวอย่างของ vad-bert)
V_MIN, V_MAX = 1.0, 5.0
A_MIN, A_MAX = 1.0, 5.0

def _to_minus1_1(x: float, xmin: float = 1.0, xmax: float = 5.0) -> float:
    """แมปสเกล [xmin, xmax] -> [-1, 1]"""
    return float(2 * (x - xmin) / (xmax - xmin) - 1.0)

def get_text_VA(text: str) -> Tuple[float, float]:
    """
    รับข้อความเดียว แล้วคืนค่า (valence, arousal) แบบ normalized อยู่ในช่วง [-1, 1]
    จากโมเดล RobroKools/vad-bert
    """
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    # logits: [1, 3] = [V, A, D]
    vad = out.logits.cpu().numpy()[0]  # shape (3,)
    v_raw, a_raw, d_raw = vad.tolist()

    v_norm = _to_minus1_1(v_raw, V_MIN, V_MAX)
    a_norm = _to_minus1_1(a_raw, A_MIN, A_MAX)

    return v_norm, a_norm

# ==============================
# 3) System prompt: client & therapist
# ==============================

CLIENT_SYSTEM = """
You are a CBT therapy client talking to therapist "Luna".

- You have anxiety, guilt, and loneliness related to your life.
- Speak in a natural, first-person voice.
- Stay emotionally consistent across turns.
- Describe thoughts, feelings, and situations in 2–4 sentences per turn.
"""

CLIENT_USER_TEMPLATE_FIRST = """
Start the first message to your therapist.
Describe what has been bothering you lately (2–4 sentences).
"""

CLIENT_USER_TEMPLATE_NEXT = """
Therapist just said:
"{therapist_text}"

Continue the conversation as the client.
Describe what you think and feel now in 2–4 sentences.
"""

THERAPIST_SYSTEM_EMO = """
You are "Luna", a warm CBT therapist.

You receive for each client message:
- The raw text of what the client said.
- An estimated emotional profile from text analysis:
  - Valence: from -1 (very negative) to +1 (very positive)
  - Arousal: from -1 (very low/flat) to +1 (very activated/agitated)

Use this emotional information to:
- Adjust your empathy (e.g., acknowledge high distress when arousal is high and valence low).
- Choose questions that fit the emotional intensity.
- Still focus on CBT techniques (thoughts, evidence, alternative views).

Important:
- NEVER mention numbers or "valence/arousal" explicitly.
- Talk only in natural emotional language (e.g., "it sounds very overwhelming").
"""

THERAPIST_USER_TEMPLATE_EMO = """
Client just said:
"{client_text}"

Estimated emotion from their words:
- Valence (text): {val_t:.2f}
- Arousal (text): {aro_t:.2f}

Write your next therapist response.
Remember: use this emotional profile internally to guide your tone and focus,
but do NOT mention these scores directly.
"""

# ==============================
# 4) helper เรียก LLM
# ==============================

def chat_once(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()

# ==============================
# 5) main loop – dialogue_5 emotion-aware
# ==============================

def run_dialogue_5_emotion(
    max_turns: int = 3,
    out_path: str = "dialogue_5_full_emotion_online.jsonl",
):
    """
    1 turn = client พูด 1 ครั้ง + therapist ตอบ 1 ครั้ง
    จะรันตั้งแต่ turn=1 จนถึง max_turns
    """
    turns = []

    # ---- turn 1: client เริ่ม ----
    client_text = chat_once(CLIENT_SYSTEM, CLIENT_USER_TEMPLATE_FIRST)
    print(f"CLIENT (t=1): {client_text}\n")

    # คำนวณ VA จากข้อความ client เทิร์น 1
    val_t, aro_t = get_text_VA(client_text)

    therapist_text = chat_once(
        THERAPIST_SYSTEM_EMO,
        THERAPIST_USER_TEMPLATE_EMO.format(
            client_text=client_text,
            val_t=val_t,
            aro_t=aro_t,
        ),
    )
    print(f"THERAPIST (t=1): {therapist_text}\n")

    turns.append({
        "turn": 1,
        "client": client_text,
        "therapist": therapist_text,
        "condition": "emotion_aware_therapist",
        "val_t": val_t,
        "aro_t": aro_t,
    })

    # ---- turn 2..max_turns ----
    for t in range(2, max_turns + 1):
        # client ตอบจากคำ therapist ล่าสุด
        client_text = chat_once(
            CLIENT_SYSTEM,
            CLIENT_USER_TEMPLATE_NEXT.format(therapist_text=therapist_text),
        )
        print(f"CLIENT (t={t}): {client_text}\n")

        # VA จากข้อความ client เทิร์นนี้
        val_t, aro_t = get_text_VA(client_text)

        therapist_text = chat_once(
            THERAPIST_SYSTEM_EMO,
            THERAPIST_USER_TEMPLATE_EMO.format(
                client_text=client_text,
                val_t=val_t,
                aro_t=aro_t,
            ),
        )
        print(f"THERAPIST (t={t}): {therapist_text}\n")

        turns.append({
            "turn": t,
            "client": client_text,
            "therapist": therapist_text,
            "condition": "emotion_aware_therapist",
            "val_t": val_t,
            "aro_t": aro_t,
        })

    # เซฟ JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for item in turns:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved dialogue to {out_path}")


if __name__ == "__main__":
    run_dialogue_5_emotion(max_turns=3)
