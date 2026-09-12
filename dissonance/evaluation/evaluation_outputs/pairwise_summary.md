# Pairwise Preference Evaluation - Luna v4 (Dissonance vs Baseline / Multimodal)

**Judge1:** openai/gpt-4o (temp 0, forced choice) | **Judge2 (kappa subset):** anthropic/claude-sonnet-4.6
**Protocol:** 2-order swap consistency (win only if preferred in both orders; else tie) - Zheng et al., 2023 (arXiv:2306.05685)
**Dialogues:** 1-100 | **Backbones:** gpt4o-mini, claude, qwen, deepseek

## Results (win = dissonance)

| Backbone | Pair | W | T | L | Net win-rate | Sign-test p | Pos. consistency | mean Dlen | r(Dlen,outcome) | Errors |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| gpt4o-mini | dissonance_vs_baseline | 61 | 14 | 25 | 0.680 | 0.0001 | 0.86 | -25 | -0.17 | 0 |
| claude | dissonance_vs_baseline | 0 | 99 | 1 | 0.495 | 1.0000 | 0.01 | -766 | -0.01 | 0 |
| qwen | dissonance_vs_baseline | 36 | 23 | 41 | 0.475 | 0.6488 | 0.77 | +672 | +0.07 | 0 |
| deepseek | dissonance_vs_baseline | 46 | 40 | 14 | 0.660 | 0.0000 | 0.60 | +152 | +0.06 | 0 |
| gpt4o-mini | dissonance_vs_multimodal | 73 | 14 | 13 | 0.800 | 0.0000 | 0.86 | -1310 | +0.00 | 0 |
| claude | dissonance_vs_multimodal | 0 | 100 | 0 | 0.500 | 1.0000 | 0.00 | -1280 | +nan | 0 |
| qwen | dissonance_vs_multimodal | 45 | 21 | 34 | 0.555 | 0.2604 | 0.79 | -342 | -0.07 | 0 |
| deepseek | dissonance_vs_multimodal | 25 | 54 | 21 | 0.520 | 0.6587 | 0.46 | -961 | +0.12 | 0 |

## Interpretation criteria (pre-registered)

- p < 0.05 and W > L  -> dissonance superiority supported for that backbone/pair
- W ~= L (p >= 0.05)  -> no detectable difference (judge ceiling)
- p < 0.05 and L > W  -> evidence against dissonance for that backbone/pair

## Inter-judge agreement (Cohen's kappa, stratified subset, seed=42)

**Overall kappa = 0.217** (raw agreement 48.0%, n=100)

| Backbone | Pair | n | kappa | raw agreement |
|---|---|--:|--:|--:|
| claude | dissonance_vs_baseline | 13 | 0.000 | 30.8% |
| claude | dissonance_vs_multimodal | 13 | 0.000 | 7.7% |
| deepseek | dissonance_vs_baseline | 13 | 0.500 | 69.2% |
| deepseek | dissonance_vs_multimodal | 13 | 0.444 | 61.5% |
| gpt4o-mini | dissonance_vs_baseline | 12 | 0.158 | 33.3% |
| gpt4o-mini | dissonance_vs_multimodal | 12 | 0.348 | 58.3% |
| qwen | dissonance_vs_baseline | 12 | 0.600 | 75.0% |
| qwen | dissonance_vs_multimodal | 12 | 0.258 | 50.0% |

## Methodology citations

- Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 D&B. arXiv:2306.05685. [pairwise protocol, position swap, verbosity bias]
- Shi, L., Ma, C., Liang, W., Diao, X., Ma, W., & Vosoughi, S. (2025). Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge. AACL-IJCNLP 2025. arXiv:2406.07791. [positional consistency reporting]
- Kim, S., et al. (2025). MIRROR: Multimodal Cognitive Reframing Therapy for Rolling with Resistance. EMNLP 2025. arXiv:2504.13211. [rubric framework; documented length bias of GPT evaluators; expert pairwise validation]
- Young, J.E., & Beck, A.T. (1980). Cognitive Therapy Rating Scale. Beck Institute. [CTRS items: Understanding, Interpersonal Effectiveness, Collaboration, Guided Discovery, Focus, Strategy for Change]
- Haydarov, K., et al. (2025). Towards AI-Assisted Psychotherapy: Emotion-Guided Generative Interventions. EMNLP 2025. [GPT-4o judge generosity/misalignment with experts -> motivates pairwise + cross-judge kappa]
- Sun, X., et al. Beyond Text: Improving LLM's Decision Making for Robot Navigation via Vocal Cues. [winning-rate metric for vocal-cue LLM systems]

Note: transcripts are full sessions from separate dialogue runs on the same problem seed; client utterances therefore differ between sessions. Judges are blind to condition identity; vocal-cue references by multimodal/dissonance therapists are intrinsic to those conditions and cannot be fully masked (limitation).