# AGENTS.md

## Project Overview
This repository supports **Luna AI Therapist**, a thesis project focused on multimodal emotion understanding for CBT-oriented dialogue generation. The core research problem is that a client’s **spoken words** and **vocal affect** do not always align. The system therefore tries to detect **emotional dissonance** (mismatch between text emotion and speech emotion) and use that signal to generate safer, more empathetic, and more context-aware therapist responses.

The current thesis direction is:
- Compare **baseline** vs **dissonance-aware** therapist generation.
- Use **text + speech** as the main modalities.
- Represent emotion mainly through **Valence-Arousal (VA)** style signals.
- Treat mismatch as a proxy for **hidden, masked, suppressed, or not-yet-verbalized affect**.
- Frame the system for **therapeutic exploration**, not lie detection.

---

## **Rules**
- **DO NOT** execute any .py/.ipynb codes after you finished modifying or generating it --> READ/WRITE only.
- **ALWAYS** back up files as `.bak` before editing.
- **DO NOT** run any git command, I will be the one who do it.

---

## Current Research Scope

### Main task
Generate therapist responses for CBT-style dialogue while using multimodal emotional signals to identify moments where the client may be saying one thing but feeling another.

### Main comparison in the proposal
1. **Baseline model**
   - Generates therapist replies without explicit emotional dissonance reasoning.
   - Usually relies on the textual content and ordinary prompting.

2. **Emotion model**
   - Context-Only with VA added: Receives pure text transcripts of the dialogue history (e.g., previous turns + current client utterance) + valence and arousal values to guide the LLM to understand the current emotional state more clearly.
   - To establish the "Text-Only with the addition of VA values Performance Ceiling".
   - This represents how effective of LLMs response via added VA values to LLMs that could help in more emphatic feel for the client's cognitive state compared to the baseline model.

3. **Multimodal Model (Vocal-Aware LLM Framework)**
   - Acts as the central reasoning engine that integrates both semantic and acoustic information.
   - Instead of processing audio separately with a complex neural network (like LSTM), we adopt a "Feature Injection" strategy.
   - Process:
      1. Extraction: Raw audio is processed to extract key prosodic features (Pitch, Loudness, Speaking Rate).
      2. Transformation: These numerical features are converted into Contextual Descriptors (e.g., "High Pitch", "Rapid Speech", or normalized values).
      3. Fusion: These acoustic descriptors are appended directly into the LLM's System Prompt alongside the dialogue text.
   - The LLM analyzes the Cognitive State by cross-referencing:
      1. What was said (Text/Semantics)
      2. How it was said (Vocal cues)
   - Result: A holistic prediction that captures emotional nuances invisible to text-only models.

4. **Dissonance Model (Dissonance-Aware LLM Framework)**
   - Acts as the central reasoning engine that integrates both semantic and acoustic information with details of cognitive dissonance measured from delta of VA between modalities (text, speech).
   - Process:
      1. Extraction: Raw audio is processed to extract key prosodic features (Pitch, Loudness, Speaking Rate).
      2. Transformation: These numerical features are converted into Contextual Descriptors (e.g., "High Pitch", "Rapid Speech", or normalized values).
      3. Fusion: These acoustic descriptors are appended directly into the LLM's System Prompt alongside the dialogue text.
   - The LLM analyzes the Cognitive State by cross-referencing:
      1. What was said (Text/Semantics)
      2. How it was said (Vocal cues)
      3. Why the expression tone differs from word meaning (Dissonance aware cues)
   - Result: A holistic prediction that captures emotional nuances invisible to text-only models and original text+speech multimodal model.

### Deliverable focus for the current proposal
- PPT comparing **baseline vs emotional dissonance method**.
- Include evaluation results already completed.
- Explain motivation and method differences clearly.
- New ideas such as disentanglement and adaptive fusion are currently **future work / next plan**, not the main implemented contribution.

---

## Core Method: Original Dissonance Pipeline

### High-level logic
1. Receive a user utterance with **speech + text**.
2. Extract emotional representation from **text**.
3. Extract emotional representation from **audio/speech**.
4. Convert both into comparable **VA-style features**.
5. Compute a dissonance score from the difference.
6. If the score exceeds a fixed threshold (commonly **0.5**), mark the turn as emotionally dissonant.
7. Use that flag to alter the therapist prompt.

### Simplified formula
Typical original logic:
- `dissonance = |V_speech - V_text| + |A_speech - A_text|`
- if `dissonance > 0.5` -> treat as important mismatch

### Interpretation
This does **not** mean the client is “lying.”
It means:
- the words and tone may be misaligned,
- the client may be minimizing distress,
- the client may be masking affect,
- or the client may be expressing a more complex inner state than text alone reveals.

---

## Terminology Used in This Thesis

### Emotional dissonance
Mismatch between the emotion inferred from text and the emotion inferred from speech.

### HEI pattern
A compact internal shorthand used in the project for the speech-text mismatch pattern that helps identify hidden affect or emotionally incongruent moments.

### HAD
A project-level framing for using hidden affect / emotional dissonance as a reasoning layer before therapist response generation.

### Shared T+A
A case where **text and audio** point in the same emotional direction.
Examples:
- positive text + positive/relaxed voice
- negative text + distressed/tired voice
This is the non-dissonant or low-dissonance case.

### Hidden feeling / masked affect
A practical therapeutic interpretation of dissonance: the client’s explicit wording may not fully reveal the affect signaled by prosody.

---

## Model and Data Components

### Text-side emotion
The project uses text analysis to estimate emotional values from the client’s utterance. This is usually represented in VA-like form rather than only discrete labels.

### Speech-side emotion
The project uses speech emotion models such as **WavLM / wav2vec-like pipelines** to estimate affect from audio.

### Fusion style in the original implementation
The original system does **not** use sophisticated adaptive fusion. It mainly:
- computes text and speech affect separately,
- compares them directly,
- uses a threshold rule,
- and triggers prompt changes when mismatch is high.

---

## Response Generation Logic

### Baseline prompting
The therapist LLM receives the dialogue context and generates a CBT-oriented response without explicit mismatch reasoning.

### Dissonance-aware prompting
When dissonance is detected, the prompt is upgraded so the model knows:
- speech and text may be emotionally inconsistent,
- the client may be uncomfortable, suppressing emotion, or not saying everything directly,
- the next response should be more empathic and exploratory.

### Safe therapeutic stance
The model should:
- explore gently,
- validate emotion,
- invite elaboration,
- avoid accusations,
- avoid claiming certainty about hidden motives,
- avoid saying the client is deceptive.

Preferred style:
- “I sense there may be some discomfort here.”
- “You said you are okay, but it sounds like this may still feel heavy.”
- “Would you like to tell me more about what made that difficult?”

Avoid:
- “You are lying.”
- “You are hiding the truth.”
- “Your statement is false.”

---

## Evaluation Direction So Far

The proposal and existing notebooks compare:
- **baseline therapist generation**
- **emotion-online / emotion-aware generation**
- **dissonance-aware generation**

The main purpose of evaluation is to show whether dissonance-aware prompting improves the quality of therapist responses relative to the baseline.

### Typical evaluation goals
- better empathy,
- better emotional attunement,
- better alignment with hidden distress,
- better intervention quality,
- better probing at difficult moments.

### Important note
For proposal presentation, the evaluation that is already complete should be emphasized. Future methods should be presented only as **next-step plans**.

---

## Proposal Narrative

### Why baseline is not enough
Baseline assumes the user’s wording directly reflects inner feeling. In therapy, this is often false because people may:
- minimize,
- avoid embarrassment,
- protect themselves,
- use emotionally flat language while sounding distressed.

### Why dissonance helps
If text says “I’m okay” but voice sounds tense, sad, or highly aroused, the system can detect that a deeper therapeutic response may be needed.

### Main thesis claim
A **multimodal emotional dissonance signal** can help a CBT dialogue model respond more safely and more appropriately than text-only or baseline prompting.

---

## Current “Next Step” Ideas
These are important, but currently treated as **future work**, not the primary implemented thesis contribution.

### 1. Multimodal Information Disentangled Transformer
Planned inspiration: split multimodal affect into:
- **shared** information: what text and audio agree on,
- **unique** information: what only one modality reveals,
- **synergy**: what emerges only when both are considered together.

Reason this matters:
- not every mismatch means hidden distress,
- some cases may reflect sarcasm, playfulness, emphasis, or other composite affect,
- disentanglement may reduce false alarms.

For proposal wording:
- only mention the **shared / unique / synergy** idea on top of T+A,
- do not claim it is already implemented unless code and evaluation are complete.

### 2. Adaptive Fusion over HAD / HEI
Planned idea:
- instead of using a fixed threshold like `0.5` for every turn,
- adjust sensitivity depending on session context.

Examples of future adaptive logic:
- more sensitive after a strong emotional shift,
- less sensitive during noisy early-session warm-up,
- different weighting of text vs voice depending on context.

Short explanation:
Adaptive fusion extends static-threshold dissonance by making the system’s mismatch sensitivity change across the dialogue instead of applying one rigid rule to all turns.

### 3. SEPSIS-inspired Dissonance Reasoner
A newer paper on omission/deception-style text reasoning inspired a possible extension.
This should be adapted **carefully** and **therapeutically**, not as forensic lie detection.

Potential use:
- run the SEPSIS-style text analysis **only when dissonance is triggered**,
- analyze the text for structural or psychological clues such as:
  - possible intent (e.g., self-protection, avoiding embarrassment),
  - omitted 5W details (who, what, when, where, why),
  - distortion/speculation/opinion-like phrasing.

Therapeutic reframing:
- not “catching lies,”
- but understanding **why affect may be masked** and **where to probe gently next**.

Recommended pipeline:
1. HEI / VA mismatch detects a critical turn.
2. SEPSIS-inspired text reasoning explains the mismatch.
3. Therapist LLM generates a safer CBT exploration response.

---

## Relation to Criminology / Deception Theory
There was an advisor-inspired discussion around criminology and deception detection as conceptual inspiration.
The safe and acceptable position for this thesis is:

### What can be borrowed
- the idea that **single cues are weak**, but multiple cues together can be useful,
- the idea that **verbal + nonverbal** information should be considered together,
- the idea that contradictions and omissions can guide further questioning.

### What should NOT be claimed
- that Luna is a lie detector,
- that the system can determine truth vs falsehood,
- that therapeutic clients should be judged using forensic deception tools.

### Safe framing
Use terms like:
- emotional dissonance,
- hidden affect,
- masked emotion,
- incomplete emotional disclosure,
- defensive or avoidant expression.

Avoid terms like:
- lie detection,
- deception classification,
- truth verification.

---

## Mermaid Framework Summary
For slides and figures, the original dissonance framework should usually be represented as:

1. User utterance (speech + text)
2. Text transcription / text emotion model
3. Audio signal / speech emotion model
4. Text VA and speech VA
5. Dissonance computation
6. Threshold decision (`> 0.5` in the original version)
7. Baseline prompt vs dissonance-aware prompt
8. Therapist reply

Important: for the **original method figure**, do not mix in:
- adaptive fusion,
- SEPSIS reasoning,
- shared/unique/synergy disentanglement,
- or other future-work modules.

Those should be drawn in separate “future framework” diagrams.

---

## Practical Guidelines for Coding Assistants
Any coding agent working in this repository should follow these rules.

### Prioritize the current thesis contribution
If asked to help with the main proposal or current experiments, focus on:
- baseline vs dissonance-aware comparison,
- evaluation artifacts already completed,
- original threshold-based dissonance logic,
- clear explanation of methodology.

Do not automatically refactor everything into future-work architectures unless explicitly asked.

### Keep terminology consistent
Prefer:
- emotional dissonance
- hidden affect
- multimodal mismatch
- text-speech incongruence
- therapist exploration

Use cautiously:
- omission reasoning
- self-protective intent
- masked emotion

Avoid unless explicitly discussing related work:
- lie detection
- deception classifier
- forensic inference

### When generating prompts
Prompts should encourage:
- empathy,
- reflection,
- clarification,
- CBT-compatible exploration,
- non-judgmental follow-up.

### When adding new modules
If implementing a future extension, clearly state whether it belongs to:
- **implemented current method**, or
- **future / experimental branch**.

### When writing figures or slides
Separate these cleanly:
- **Current implemented framework**
- **Next-step research plan**

Do not merge them into one diagram unless explicitly requested.

---

## Recommended Repository Labels / Branch Meanings
If the repository needs conceptual organization, these labels are useful:

- `baseline/` -> baseline therapist generation
- `emotion/` -> emotion-aware but not full dissonance
- `dissonance/` -> original threshold-based dissonance pipeline
- `future_disentangle/` -> shared/unique/synergy ideas
- `future_adaptive/` -> adaptive fusion / adaptive threshold logic
- `future_sepsis/` -> SEPSIS-inspired explanatory reasoning after dissonance trigger
- `evaluation/` -> metrics, notebooks, analysis, plots, tables
- `slides/` -> proposal and defense presentation materials

Noted that most of my current latest works like script files are located within dissonance directory folder: C:\Luna-AI-Therapist\dissonance>

---

## What an Ideal Assistant Should Remember
A good agent working on this thesis should remember the following core message:

> Luna AI Therapist is a CBT-oriented multimodal dialogue system that compares emotional signals from text and speech to detect emotional dissonance, then uses that mismatch to guide more empathetic and context-aware therapist responses.

And the second key message:

> The currently implemented contribution is the original threshold-based dissonance-aware framework; disentanglement, adaptive fusion, and SEPSIS-style reasoning are promising future extensions and should be presented as such unless fully implemented and evaluated.

---

## Short Version for Quick Context
If an assistant only reads one section, read this:

- Thesis: **Luna AI Therapist**
- Domain: **multimodal CBT dialogue generation**
- Main idea: compare **text emotion vs speech emotion**
- Key signal: **emotional dissonance / hidden affect**
- Original rule: threshold-based mismatch, often `> 0.5`
- Main comparison: **baseline vs dissonance-aware**
- Output goal: better therapeutic response quality
- Safe framing: **exploration of masked affect**, not lie detection
- Future work: **shared/unique/synergy disentanglement**, **adaptive fusion**, **SEPSIS-inspired reasoning**
