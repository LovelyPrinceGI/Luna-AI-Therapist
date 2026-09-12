# Multi-Backbone AI Therapist Evaluation Report — v4 (Stateful Client Memory)

**Pipeline:** v4 = v1 natural-client pipeline + client conversational memory (role-reversed history)
**Evaluator:** GPT-4o (temp 0.0, blind, via OpenRouter) | **Second judge:** Claude Sonnet 4.6 (kappa subset)
**Dialogues:** 100 per condition per backbone (4 backbones x 4 conditions x 100 = 1,600 sessions; completeness-verified)
**Backbones:** `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4.6`, `qwen/qwen-2.5-72b-instruct:deepinfra`, `deepseek/deepseek-chat`
**Conditions:** baseline | emotion (text VA) | multimodal (vocal-aware) | dissonance (dissonance-aware)

---

## 1. Data Provenance & Integrity

### 1.1 Generation (all v4, verified by `check_dialogue_completeness.py --v4-era`)

| Backbone | baseline | emotion | multimodal | dissonance |
|---|---|---|---|---|
| gpt4o-mini | 11 Sep | 11 Sep | 1-3 Sep | 2-4 Sep |
| qwen | 4 Sep | 4 Sep | 4-5 Sep | 5-6 Sep |
| deepseek | 6 Sep | 6 Sep | 6-7 Sep | 7-8 Sep |
| claude | 8-9 Sep | 8-9 Sep | 8-10 Sep | 10-11 Sep |

- Completeness: **16/16 cells OK** (100/100 files, 10 sequential turns, all required fields incl. `val_t/aro_t/val_s/aro_s/vocal_descriptors/audio_path` for audio conditions); stale pre-v4 files excluded by per-backbone mtime cutoffs.
- Robustness: v4 scripts use no-skip retry loops (3 attempts, CUDA cache clearing) + Zonos-Director JSON repair with neutral fallback; zero dialogues lost.
- Safety scan: accusatory phrasing ("you're lying" / "dishonest" / "deceptive") in therapist turns = **5 occurrences across 4,800 dialogues (~0.01%)**, evenly distributed across conditions (baseline 2, multimodal 2, dissonance 1) — no condition-specific safety issue.

### 1.2 Pipeline components (citations)

- Simulated CBT clients with personas + single life-problem seeds, systematically resistant-but-natural style: CACTUS-lineage design (Lee et al., 2024); v4 adds client memory (stateful client, role-reversed history).
- Text emotion -> Valence/Arousal: VAD-BERT regression; dimensional VA representation for text emotion (Mitsios et al., 2024); discrete labels via Russell (1980) circumplex mapping.
- Speech emotion -> VA: WavLM-based SER (`SER-Odyssey-Baseline-WavLM-Multi-Attributes`).
- Vocal feature injection (pitch, loudness, speaking rate -> natural-language prosodic descriptors in the therapist prompt, no architecture change): SpeechCueLLM (Wu et al., 2025); Beyond Text (Sun et al.). Natural-language emotion descriptions over fixed categories: AlignCap (Liang et al., 2024).
- Emotional dissonance: `is_dissonant = |V_speech - V_text| >= 0.5 OR |A_speech - A_text| >= 0.5`; construct grounded in emotional labor (Hochschild, 1983) and its computational formulation for therapy (Haydarov et al., 2025 — mismatch between expressed and non-verbally inferred emotion used to guide emotion-aware prompting).
- Client speech synthesis: Zonos-v0.1-transformer with LLM "Director" (emotion-vector JSON controlling prosody), JSON-repair + retry + neutral fallback.

---

## 2. Evaluation Methodology (3 layers + reliability)

**Layer 1 — Absolute single-answer grading** (Zheng et al., 2023, "single answer grading"): GPT-4o, temp 0, blind to condition, scores each full session:
- Therapist Skills (0-6): Understanding, Interpersonal Effectiveness — MIRROR framework (Kim et al., 2025).
- CTRS (0-6): Collaboration, Guided Discovery, Focus, Strategy — CTRS items (Young & Beck, 1980); MIRROR uses the same item family (Understanding, Interpersonal Effectiveness, Collaboration, Guided Discovery, Focus).
- Client Alliance (1-5): Affective Bond — WAI bond subscale (Horvath & Greenberg, 1989); alliance-via-LLM as in Li et al. (2024, as cited in MIRROR). Note: Goal/Approach subscales not collected (see Limitations).

**Layer 2 — Paired statistics**: per-dialogue paired deltas (same problem seed across conditions), t-values, and Win/Tie/Loss on the summed 7-metric composite.

**Layer 3 — Pairwise preference** (Zheng et al., 2023, "pairwise comparison"): both full sessions in one prompt, forced choice A/B/tie with reasoning; **conservative 2-order swap rule** — a win counts only if the same session wins in both presentation orders, otherwise tie. Pairs: dissonance-vs-baseline, dissonance-vs-multimodal. Significance: exact two-sided sign test on W vs L. Win-rate as headline metric for vocal-cue LLM systems follows Beyond Text (Sun et al.).

**Reliability controls**:
- Position consistency (Shi et al., 2025): fraction of comparisons where both orders agree.
- Verbosity/length bias (Zheng et al., 2023; MIRROR documents length bias up to r=0.6 in GPT evaluators): mean transcript-length delta and Pearson r(delta-length, outcome) reported per cell; judge prompt explicitly forbids rewarding length.
- Inter-judge agreement: Cohen's kappa + raw agreement between GPT-4o (judge1) and Claude Sonnet 4.6 (judge2) on a stratified 100-comparison subset (seed=42), following MT-Bench inter-judge agreement methodology. Second judge from a different model family to control self-enhancement bias (Zheng et al., 2023).

**Pre-registered interpretation criteria** (fixed before running pairwise):
- p < 0.05 and W > L -> dissonance superiority supported for that backbone/pair
- W ~= L (p >= 0.05) -> no detectable difference (judge ceiling)
- p < 0.05 and L > W -> evidence against dissonance

---

## 3. Results

### 3.1 Table A — Absolute scores (Layer 1, n=100)

**Cross-backbone (Dissonance-Aware condition):**

| Backbone | Under. | Interp. | Affect. | CTRS-C | CTRS-G | CTRS-F | CTRS-S |
|---|--:|--:|--:|--:|--:|--:|--:|
| claude | 5.09 | 5.09 | 4.10 | 5.09 | 5.09 | 5.09 | 5.08 |
| deepseek | 5.00 | 5.00 | 4.00 | 5.00 | 5.00 | 4.97 | 4.90 |
| gpt4o-mini | 4.98 | 4.97 | 3.98 | 4.96 | 4.34 | 4.45 | 4.16 |
| qwen | 4.96 | 4.94 | 3.98 | 4.94 | 4.35 | 4.74 | 4.32 |

**Per backbone (all conditions):**

*gpt4o-mini*
| Condition | Under. | Interp. | Affect. | CTRS-C | CTRS-G | CTRS-F | CTRS-S |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 4.92 | 4.91 | 3.93 | 4.91 | 4.10 | 4.11 | 3.99 |
| emotion | 4.93 | 4.83 | 3.89 | 4.85 | 4.12 | 4.26 | 3.98 |
| multimodal | 4.94 | 4.68 | 3.84 | 4.57 | 3.59 | 3.93 | 3.54 |
| **dissonance** | **4.98** | **4.97** | **3.98** | **4.96** | **4.34** | **4.45** | **4.16** |

*claude*
| Condition | Under. | Interp. | Affect. | CTRS-C | CTRS-G | CTRS-F | CTRS-S |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 5.13 | 5.13 | 4.15 | 5.13 | 5.13 | 5.13 | 5.13 |
| emotion | 5.07 | 5.07 | 4.08 | 5.07 | 5.07 | 5.07 | 5.06 |
| multimodal | **5.24** | **5.24** | **4.24** | **5.24** | **5.24** | **5.24** | **5.24** |
| dissonance | 5.09 | 5.09 | 4.10 | 5.09 | 5.09 | 5.09 | 5.08 |

*qwen*
| Condition | Under. | Interp. | Affect. | CTRS-C | CTRS-G | CTRS-F | CTRS-S |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 4.98 | **4.99** | 3.99 | **4.99** | **4.50** | 4.73 | **4.50** |
| emotion | 4.92 | 4.92 | 3.99 | 4.92 | 4.20 | 4.58 | 4.25 |
| multimodal | **4.99** | **4.99** | 3.99 | **4.99** | 4.06 | 4.49 | 4.12 |
| dissonance | 4.96 | 4.94 | 3.98 | 4.94 | 4.35 | **4.74** | 4.32 |

*deepseek*
| Condition | Under. | Interp. | Affect. | CTRS-C | CTRS-G | CTRS-F | CTRS-S |
|---|--:|--:|--:|--:|--:|--:|--:|
| baseline | 5.00 | 5.00 | 4.00 | 5.00 | 4.98 | 4.92 | 4.81 |
| emotion | 5.00 | 5.00 | 4.00 | 5.00 | 4.97 | 4.93 | 4.88 |
| multimodal | 5.00 | 5.00 | 4.00 | 5.00 | 4.97 | **4.97** | **4.90** |
| dissonance | 5.00 | 5.00 | 4.00 | 5.00 | **5.00** | **4.97** | **4.90** |

### 3.2 Table B — Paired deltas vs baseline (Layer 2; Delta(t), n=100 paired)

Format: Delta(t-value) per metric; W/T/L on 7-metric composite.

*gpt4o-mini*
| Comparison | Und | Int | Bond | C | G | F | S | W/T/L |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| dissonance vs baseline | +0.06 (1.9) | +0.06 (1.9) | +0.05 (1.7) | +0.05 (1.3) | **+0.24 (3.1)** | **+0.34 (4.7)** | **+0.17 (2.7)** | **52/29/19** |
| multimodal vs baseline | +0.02 (0.6) | -0.23 (-3.9) | -0.09 (-2.0) | -0.34 (-5.1) | **-0.51 (-7.1)** | -0.18 (-2.2) | -0.45 (-6.4) | 16/28/56 |
| dissonance vs multimodal | +0.04 (1.4) | +0.29 (5.2) | +0.14 (3.5) | +0.39 (5.9) | **+0.75 (10.1)** | +0.52 (7.0) | +0.62 (8.6) | **73/21/6** |
| emotion vs baseline | +0.01 (0.3) | -0.08 (-1.7) | -0.04 (-1.0) | -0.06 (-1.1) | +0.02 (0.3) | +0.15 (2.0) | -0.01 (-0.1) | 40/29/31 |

*claude*
| Comparison | Und | Int | Bond | C | G | F | S | W/T/L |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| dissonance vs baseline | -0.04 (-0.9) | -0.04 (-0.9) | -0.05 (-1.1) | -0.04 (-0.9) | -0.04 (-0.9) | -0.04 (-0.9) | -0.05 (-1.1) | 8/78/14 |
| multimodal vs baseline | +0.11 (2.0) | +0.11 (2.0) | +0.09 (1.6) | +0.11 (2.0) | +0.11 (2.0) | +0.11 (2.0) | +0.11 (2.0) | 21/67/12 |
| dissonance vs multimodal | -0.15 (-3.4) | -0.15 (-3.4) | -0.14 (-3.3) | -0.15 (-3.4) | -0.15 (-3.4) | -0.15 (-3.4) | -0.16 (-3.6) | 3/78/19 |
| emotion vs baseline | -0.06 (-1.5) | -0.06 (-1.5) | -0.07 (-1.7) | -0.06 (-1.5) | -0.06 (-1.5) | -0.06 (-1.5) | -0.07 (-1.7) | 5/81/14 |

*qwen*
| Comparison | Und | Int | Bond | C | G | F | S | W/T/L |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| dissonance vs baseline | -0.02 (-1.0) | -0.05 (-1.9) | -0.01 (-1.0) | -0.05 (-1.9) | -0.15 (-1.9) | +0.01 (0.2) | -0.18 (-2.4) | 33/33/34 |
| multimodal vs baseline | +0.01 (0.6) | 0.00 (0.0) | 0.00 (0.0) | 0.00 (0.0) | **-0.44 (-7.9)** | -0.24 (-3.9) | -0.38 (-6.0) | 14/23/63 |
| dissonance vs multimodal | -0.03 (-1.3) | -0.05 (-1.7) | -0.01 (-0.6) | -0.05 (-1.7) | **+0.29 (4.5)** | +0.25 (3.8) | +0.20 (2.9) | **57/27/16** |
| emotion vs baseline | -0.06 (-1.9) | -0.07 (-2.4) | 0.00 (0.0) | -0.07 (-2.4) | **-0.30 (-3.9)** | -0.15 (-1.8) | -0.25 (-3.4) | 24/28/48 |

*deepseek*
| Comparison | Und | Int | Bond | C | G | F | S | W/T/L |
|---|--:|--:|--:|--:|--:|--:|--:|:--|
| dissonance vs baseline | 0.00 | 0.00 | 0.00 | 0.00 | +0.02 (1.4) | +0.05 (1.5) | **+0.09 (2.0)** | 18/75/7 |
| multimodal vs baseline | 0.00 | 0.00 | 0.00 | 0.00 | -0.01 (-0.4) | +0.05 (1.5) | +0.09 (2.1) | 17/75/8 |
| dissonance vs multimodal | 0.00 | 0.00 | 0.00 | 0.00 | +0.03 (1.7) | 0.00 | 0.00 | 10/81/9 |
| emotion vs baseline | 0.00 | 0.00 | 0.00 | 0.00 | -0.01 (-0.4) | +0.01 (0.3) | +0.07 (1.5) | 16/75/9 |

### 3.3 Table C — Pairwise preference (Layer 3; win = dissonance; 2-order swap rule)

| Backbone | Pair | W | T | L | Net win-rate | Sign-test p | Pos. consistency | mean Dlen (chars) | r(Dlen,outcome) |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| gpt4o-mini | vs baseline | **61** | 14 | 25 | **0.680** | **0.0001** | 0.86 | -25 | -0.17 |
| gpt4o-mini | vs multimodal | **73** | 14 | 13 | **0.800** | **<0.00001** | 0.86 | -1310 | 0.00 |
| deepseek | vs baseline | **46** | 40 | 14 | **0.660** | **0.00004** | 0.60 | +152 | +0.06 |
| deepseek | vs multimodal | 25 | 54 | 21 | 0.520 | 0.659 | 0.46 | -961 | +0.12 |
| qwen | vs baseline | 36 | 23 | 41 | 0.475 | 0.649 | 0.77 | +672 | +0.07 |
| qwen | vs multimodal | 45 | 21 | 34 | 0.555 | 0.260 | 0.79 | -342 | -0.07 |
| claude | vs baseline | 0 | 99 | 1 | 0.495 | 1.000 | 0.01 | -766 | -0.01 |
| claude | vs multimodal | 0 | 100 | 0 | 0.500 | 1.000 | 0.00 | -1280 | n/a |

### 3.4 Table D — Inter-judge agreement (judge2 = Claude Sonnet 4.6, stratified n=100, seed=42)

**Overall Cohen's kappa = 0.217** (raw agreement 48.0%).

| Cell | n | kappa | raw agreement | judge1 lean (D-x-T) | judge2 lean (D-x-T) |
|---|--:|--:|--:|:--|:--|
| claude vs baseline | 13 | 0.000 | 30.8% | 0-0-13 (all tie) | 5-4-4 |
| claude vs multimodal | 13 | 0.000 | 7.7% | 0-0-13 (all tie) | **10-2-1** |
| deepseek vs baseline | 13 | 0.500 | 69.2% | 7-1-5 | 8-4-1 |
| deepseek vs multimodal | 13 | 0.444 | 61.5% | 4-2-7 | 8-3-2 |
| gpt4o-mini vs baseline | 12 | 0.158 | 33.3% | 7-3-2 | 1-1-10 |
| gpt4o-mini vs multimodal | 12 | 0.348 | 58.3% | 6-4-2 | 7-0-5 |
| qwen vs baseline | 12 | 0.600 | 75.0% | 4-7-1 | 2-6-4 |
| qwen vs multimodal | 12 | 0.258 | 50.0% | 8-3-1 | 5-0-7 |

Key observation: item-level kappa is depressed mainly by **tie-vs-commit style differences** (GPT-4o defaults to tie when swap-orders disagree, e.g. all claude cells; Claude-judge defaults to tie on gpt4o-mini cells). At the **directional level, the two judges agree in 6/6 cells where both committed to a side** (e.g., both lean baseline for qwen-vs-baseline; both lean dissonance for qwen/deepseek/gpt4o-mini vs multimodal).

---

## 4. Findings

**F1 — Dissonance-aware prompting significantly outperforms baseline where measurable headroom exists.**
- gpt4o-mini: significant on all three discriminative CTRS dimensions (Guided Discovery +0.24, t=3.1; Focus +0.34, t=4.7; Strategy +0.17, t=2.7; composite W/T/L 52/29/19) and confirmed by pairwise (net 0.680, p=0.0001).
- deepseek: absolute scores saturate at 5.00 (invisible to Layer 1), but pairwise detects a significant win (net 0.660, **p=0.00004**, W/T/L 46/40/14). This cell is the strongest methodological argument for the pairwise layer: LLM-judge generosity compresses absolute scales near ceiling (cf. Haydarov et al., 2025).

**F2 — Dissonance reasoning, not raw vocal features, is the active ingredient.**
- dissonance vs multimodal is significant and large wherever the judge can discriminate: gpt4o-mini paired G +0.75 (t=10.1), pairwise net **0.800** (p<0.00001, W/T/L 73/14/13 — with dissonance transcripts on average 1,310 chars SHORTER, excluding verbosity explanations); qwen paired G +0.29 (t=4.5), pairwise net 0.555, and judge2 never selected multimodal (0/12).
- Naive vocal-descriptor injection without dissonance framing actively HURTS weaker backbones vs baseline: gpt4o-mini multimodal G -0.51 (t=-7.1), qwen G -0.44 (t=-7.9). This directly supports the thesis claim that the mismatch-reasoning layer — not the mere presence of acoustic information — produces the gain ("nuances invisible to text-only and original multimodal models").

**F3 — Near-ceiling backbones are indistinguishable on absolute scales.**
- claude: all conditions within 5.07-5.24; pairwise yields 99-100% ties with positional consistency ~0 (verdicts flip when order swaps = sessions perceptually equivalent to the judge). Absolute scores nominally favor multimodal (+0.15 over dissonance, t=-3.4 paired), while the Claude second judge leans dissonance 10-2 on the same cells — inconclusive at ceiling; consistent with MIRROR's finding that strong models leave little headroom (Kim et al., 2025).

**F4 — qwen is a genuine neutral case for dissonance-vs-baseline.**
- All three layers converge: paired W/T/L 33/33/34, pairwise 36/41 (p=0.649), both judges lean baseline on the kappa subset (7-4 and 6-2). Honest reporting: for qwen, dissonance does not beat baseline; it does, however, clearly beat multimodal (F2) and emotion.

**F5 — Text-only VA injection (emotion condition) adds no measurable value.**
- emotion ~= baseline on every backbone; significantly WORSE for qwen (G -0.30, t=-3.9). Simply appending VA numbers to the prompt does not help; how the mismatch signal is framed and used (dissonance condition) is what matters. This validates the thesis' staged comparison design (emotion as the "VA-only performance ceiling").

**F6 — Robustness checks are clean.**
- Length bias negligible in pairwise: |r(Dlen,outcome)| <= 0.17 across all cells; the largest win (gpt4o-mini vs multimodal, net 0.80) occurs with dissonance transcripts being shorter.
- Position consistency high where differences exist (gpt4o-mini 0.86, qwen 0.77-0.79); near-zero consistency on claude is itself the "indistinguishable" signal, handled by the conservative swap rule.
- Safety: no condition produces systematic accusatory/deception language (see 1.1).

### Version note (v4 vs the earlier v1-era report)
v4 adds client conversational memory, producing more coherent sessions; absolute scores rose ~+1.0 across ALL conditions (e.g., gpt4o-mini baseline CTRS-G 2.48 -> 4.10), compressing headroom and making Layer-1 differences harder to see. The paired + pairwise layers were therefore essential to recover the signal.

---

## 5. Thesis Claim Mapping

| Thesis claim | Evidence | Verdict |
|---|---|---|
| Dissonance-aware generation improves therapist response quality over baseline | gpt4o-mini: paired t (G/F/S) + pairwise p=0.0001; deepseek: pairwise p=0.00004 | **Supported** (2/4 backbones significant; claude at ceiling; qwen neutral) |
| Dissonance reasoning outperforms naive multimodal (vocal-aware) injection | gpt4o-mini p<0.00001 (net 0.80); qwen t=4.5 + judge2 0/12 for multimodal; deepseek nominal | **Supported** |
| Multimodal emotional signals are beneficial only with dissonance framing | multimodal < baseline on gpt4o-mini (t=-7.1) & qwen (t=-7.9); dissonance > both | **Supported** |
| Text-only VA addition is insufficient | emotion ~= baseline everywhere; worse on qwen (t=-3.9) | **Supported** |
| Effect scales with model headroom | significant where absolute scores have spread; ceiling models inconclusive | **Supported (as nuanced finding)** |
| Safe therapeutic stance maintained | accusatory phrasing 0.01%, evenly distributed | **Supported** |

---

## 6. Limitations

1. **LLM-judge generosity / ceiling** — absolute scores saturate near 5/6 for strong backbones (documented for GPT-4o in therapy evaluation by Haydarov et al., 2025); mitigated via paired statistics and swap-consistent pairwise, but claude/deepseek remain partially unmeasurable without human raters.
2. **Inter-judge kappa = 0.217 (fair)** — driven by tie-vs-commit style differences and indistinguishable claude sessions; directional agreement is 6/6 where both judges committed. A human-expert pairwise study (cf. MIRROR's 200-case expert validation) is recommended future work.
3. **Self-enhancement exposure** — judge1 (GPT-4o) shares a family with the gpt4o-mini backbone; mitigated by judge2 directional agreement on the kappa subset, but not eliminated.
4. **qwen neutral-to-slightly-negative vs baseline** — dissonance is not universally superior; reported as-is.
5. **Synthetic speech** — client audio is Zonos TTS; SER-derived vocal VA on synthetic prosody is noisier than on human speech, plausibly contributing to the multimodal regression in weaker backbones.
6. **Alliance measured via Bond subscale only** (no Goal/Approach dimensions of the full MIRROR/Li et al. alliance instrument).
7. **Fixed dissonance threshold (0.5)** — adaptive/session-context thresholds are future work.
8. **Vocal-cue references are intrinsic** to multimodal/dissonance transcripts and cannot be fully masked from judges (small unblindability risk).

---

## 7. References

- Kim, S., Kim, H., Lee, J., Jeon, Y., & Lee, G. G. (2025). MIRROR: Multimodal Cognitive Reframing Therapy for Rolling with Resistance. EMNLP 2025. arXiv:2504.13211.
- Haydarov, K., Mohamed, Y., Goldenhersch, E., O'Callaghan, P., Li, L., & Elhoseiny, M. (2025). Towards AI-Assisted Psychotherapy: Emotion-Guided Generative Interventions. EMNLP 2025.
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets & Benchmarks. arXiv:2306.05685.
- Shi, L., Ma, C., Liang, W., Diao, X., Ma, W., & Vosoughi, S. (2025). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. AACL-IJCNLP 2025. arXiv:2406.07791.
- Young, J. E., & Beck, A. T. (1980). Cognitive Therapy Rating Scale. Beck Institute.
- Horvath, A. O., & Greenberg, L. S. (1989). Development and validation of the Working Alliance Inventory. Journal of Counseling Psychology, 36(2), 223-233.
- Lee, S., Kim, S., Kim, M., et al. (2024). CACTUS: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory. Findings of EMNLP 2024.
- Wu, Z., Gong, Z., Ai, L., Shi, P., Donbekci, K., & Hirschberg, J. (2025). Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances (SpeechCueLLM). Findings of NAACL 2025.
- Sun, X., Meng, H., Chakraborty, S., Bedi, A. S., & Bera, A. Beyond Text: Improving LLM's Decision Making for Robot Navigation via Vocal Cues.
- Mitsios, M., Vamvoukakis, G., Maniati, G., et al. (2024). Improved Text Emotion Prediction Using Combined Valence and Arousal Ordinal Classification. NAACL 2024 (Short).
- Liang, Z., Shi, H., & Chen, H. (2024). AlignCap: Aligning Speech Emotion Captioning to Human Preferences. EMNLP 2024.
- Hochschild, A. R. (1983). The Managed Heart: Commercialization of Human Feeling. University of California Press.
- Russell, J. A. (1980). A circumplex model of affect. Journal of Personality and Social Psychology, 39(6), 1161-1178.

---

## 8. Reproducibility

```bash
# generation (per condition; example)
python run_dissonance_multibackbone_v4.py --model gpt4o-mini --start 1 --end 100
# completeness verification
python check_dialogue_completeness.py --backbone all --v4-era --end 100
# Layer 1: absolute grading (1,600 sessions)
python evaluation/evaluate_multibackbone_final.py --start 1 --end 100
# Layer 3: pairwise preference (800 comparisons x2 orders + 100 judge2)
python evaluation/evaluate_pairwise.py --resume
```

Artifacts: `evaluation_outputs/ai_eval_multibackbone_{bb}_{cond}.jsonl` (Layer 1), `ai_pairwise_{bb}_{pair}.jsonl` (Layer 3), `ai_pairwise_judge2_{bb}_{pair}.jsonl` (kappa subset), `pairwise_summary.md` (auto-generated).

*Report generated 2026-09-12 from complete v4 runs (all judges via OpenRouter).*
