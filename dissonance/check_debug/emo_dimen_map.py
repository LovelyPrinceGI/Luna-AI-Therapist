#!/usr/bin/env python3
"""
Extract valence-arousal + discrete emotion labels from multibackbone JSON logs
and render Russell's circumplex (1980) scatter maps for text cue vs speech cue.
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = Path(__file__).resolve().parent
DATA_BASE = HERE.parent / "multibackbone"

BACKBONES = ["claude", "qwen", "gpt4o-mini", "deepseek"]
CONDITIONS = ["multimodal", "dissonance"]

EMOTION_ORDER = [
    "Happy",
    "Excited",
    "Tense",
    "Angry",
    "Sad",
    "Depressed",
    "Calm",
    "Content",
    "Neutral",
]

SECTOR_ANGLES = {
    "Happy": 0.0,
    "Excited": 45.0,
    "Tense": 90.0,
    "Angry": 135.0,
    "Sad": 180.0,
    "Depressed": 225.0,
    "Calm": 270.0,
    "Content": 315.0,
}

PALETTE = {
    "Happy": "#FFD100",
    "Excited": "#FF7F0E",
    "Tense": "#9467BD",
    "Angry": "#D62728",
    "Sad": "#1F77B4",
    "Depressed": "#2E4057",
    "Calm": "#2CA02C",
    "Content": "#98DF8A",
    "Neutral": "#7F7F7F",
}

NEUTRAL_RADIUS = 0.15


def load_records():
    rows = []
    for backbone in BACKBONES:
        for condition in CONDITIONS:
            folder = DATA_BASE / backbone / condition
            if not folder.is_dir():
                continue
            for fp in sorted(folder.glob("*.json")):
                with open(fp, encoding="utf-8") as fh:
                    payload = json.load(fh)
                for rec in payload:
                    rows.append(
                        {
                            "backbone": backbone,
                            "condition": condition,
                            "file": fp.name,
                            "turn": rec.get("turn"),
                            "val_t": rec.get("val_t"),
                            "aro_t": rec.get("aro_t"),
                            "val_s": rec.get("val_s"),
                            "aro_s": rec.get("aro_s"),
                            "text_emotion": rec.get("text_emotion"),
                            "speech_emotion": rec.get("speech_emotion"),
                        }
                    )
    return pd.DataFrame(rows)


def add_circumplex(ax):
    ax.axhline(0.0, color="0.55", lw=0.9, zorder=1)
    ax.axvline(0.0, color="0.55", lw=0.9, zorder=1)
    for k in range(8):
        ang = math.radians(22.5 + 45.0 * k)
        x0, y0 = NEUTRAL_RADIUS * math.cos(ang), NEUTRAL_RADIUS * math.sin(ang)
        x1, y1 = 1.45 * math.cos(ang), 1.45 * math.sin(ang)
        ax.plot([x0, x1], [y0, y1], ls="--", lw=0.7, color="0.72", zorder=1)
    ax.add_patch(
        plt.Rectangle(
            (-NEUTRAL_RADIUS, -NEUTRAL_RADIUS),
            2 * NEUTRAL_RADIUS,
            2 * NEUTRAL_RADIUS,
            fill=False,
            ls=":",
            lw=0.9,
            edgecolor="0.45",
            zorder=2,
        )
    )
    for name, deg in SECTOR_ANGLES.items():
        ang = math.radians(deg)
        ax.text(
            0.85 * math.cos(ang),
            0.85 * math.sin(ang),
            name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="0.35",
            alpha=0.85,
            zorder=2,
        )
    ax.text(
        0.0,
        -0.20,
        "Neutral",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="0.45",
        alpha=0.9,
        zorder=2,
    )
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")


def make_plot(df, xcol, ycol, huecol, title, outpath):
    present = [e for e in EMOTION_ORDER if e in set(df[huecol])]
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    add_circumplex(ax)
    sns.scatterplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=huecol,
        hue_order=present,
        palette=PALETTE,
        s=16,
        alpha=0.5,
        linewidth=0,
        ax=ax,
        zorder=3,
    )
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"{title}  |  n = {len(df):,} turns", fontsize=13, fontweight="bold")
    ax.legend(
        title=huecol,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


def main():
    sns.set_theme(style="white")
    df = load_records()
    print(f"loaded rows: {len(df):,}")
    print(df.groupby(["condition", "backbone"]).size())

    csv_path = HERE / "emo_dimen_map_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    text_df = df.dropna(subset=["val_t", "aro_t", "text_emotion"])
    speech_df = df.dropna(subset=["val_s", "aro_s", "speech_emotion"])

    make_plot(
        text_df,
        "val_t",
        "aro_t",
        "text_emotion",
        "Text Cue - Russell's Circumplex (Valence x Arousal)",
        HERE / "emo_dimen_map_text_cue.png",
    )
    make_plot(
        speech_df,
        "val_s",
        "aro_s",
        "speech_emotion",
        "Speech Cue - Russell's Circumplex (Valence x Arousal)",
        HERE / "emo_dimen_map_speech_cue.png",
    )


if __name__ == "__main__":
    main()
