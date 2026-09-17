# Root-Cause Analysis: ทำไม Qwen Dissonance-Aware ไม่ชนะ Baseline

**วันที่วิเคราะห์:** 2026-09-16
**ข้อมูลที่ใช้:** v4 multi-backbone runs (qwen baseline/dissonance/multimodal 100 dialogues × 10 turns) + Layer-1 scores + Layer-3 pairwise judge reasoning
**แหล่งข้อมูล:**
- `dissonance/multibackbone/qwen/{baseline,dissonance,multimodal}/*.jsonl`
- `dissonance/evaluation/evaluation_outputs/ai_eval_multibackbone_qwen_{baseline,dissonance}.jsonl`
- `dissonance/evaluation/evaluation_outputs/ai_pairwise_qwen_dissonance_vs_baseline.jsonl`

---

## TL;DR (สาเหตุหลัก)

> Dissonance prompt ของ Qwen เปลี่ยน therapist ให้เป็น **"reflection loop"** — เปิด turn ด้วย
> "It sounds like..." + พูดถึงเสียง ("your voice...") + ถาม "Can you tell me more...?" วนซ้ำ
> ทุก turn โดย **ไม่ advance ไปสู่ CBT strategy** (technique / evidence / experiment / plan)
> ภายใน budget 2–3 ประโยค → judge ลงโทษที่ **Guided Discovery / Focus / Strategy**
> (คำวิจารณ์ใน 41 เคสที่แพ้: strategy ×95, discovery ×94, surface ×17 — แต่ voice ×0)
> ผลรวมจึงเป็น null (33/33/34) แบบ "ชนะเล็กน้อยจำนวนมาก + แพ้ย่อยยับไม่กี่ session (หนักสุด −14)"
>
> นี่คือ **prompt × backbone interaction** ไม่ใช่ความล้มเหลวของ dissonance mechanism:
> prompt เดียวกันบน gpt4o-mini พูดถึง voice แค่ 21% (ไม่ loop) → ชนะ baseline อย่างมีนัยสำคัญ

---

## 1. โครงสร้างผลลัพธ์: "null with catastrophic tail"

Layer-1 composite (7 metrics, เต็ม 41) per-dialogue, dissonance − baseline:

- W/T/L = **33/33/34**, mean Δ = **−0.45** (ตรงกับรายงานหลัก)
- ชนะสูงสุด: **+3** (d29, d99)
- แพ้หนักสุด:

| dialogue | baseline | dissonance | Δ | GD | Focus | Strategy |
|---|--:|--:|--:|--:|--:|--:|
| **63** | 34 | **20** | **−14** | −3 | −2 | −3 |
| 53 | 34 | 26 | −8 | −2 | −1 | −2 |
| 51 | 34 | 27 | −7 | −2 | −1 | −2 |
| 95 | 31 | 25 | −6 | −1 | −1 | −1 |

→ ค่าเฉลี่ยเป็น null เพราะการแจกแจงแบบ **bimodal**: session ที่เป็น emotion-processing
ได้ประโยชน์จาก reflective stance (+2 ถึง +3) แต่ session ที่ต้องการ problem-solving
จะ stall หนัก (−6 ถึง −14)

---

## 2. หลักฐานเชิงปริมาณ: พฤติกรรม therapist text (1,000 turns/condition)

| Pattern (regex บน therapist text) | qwen baseline | qwen dissonance | qwen multimodal | gpt4o-mini dissonance |
|---|--:|--:|--:|--:|
| กล่าวถึงเสียง: `your voice / I can hear / voice carries / voice conveys / your tone / soft and rapid` | **1.7%** | **72.5%** | 89.9% | **21.1%** |
| Formulaic opener: ขึ้นต้นด้วย `It sounds like / I can hear / It sounds / I hear` | 8.0% | **72.9%** | 88.8% | 57.5% |
| มีคำว่า `Sometimes,` (closing formula) | 2.3% | 9.9% | 6.1% | 5.4% |
| Strategy words (`exercise/technique/experiment/evidence/journal/write down/practice/small step/plan/try/reframe...`) | 51.2% | 47.8% | 52.4% | 32.4% |

Per-dialogue level (qwen):

| ตัวชี้วัด | baseline | dissonance |
|---|--:|--:|
| Session ที่ใช้ **4-word opener เดิมซ้ำ ≥ 4 ครั้ง** | **0%** | **37%** |
| Turn แรกที่มี strategy word (เฉลี่ย) | **2.9** | 3.3 |
| Session ที่ไม่มี strategy word เลย | 0% | 1% |

**ข้อสรุปสำคัญ:** prompt เดียวกันให้พฤติกรรมต่างกันลิบลับตาม backbone —
Qwen-2.5-72B (instruction-following สูง) "เล่นตาม" metadata ที่ inject ทุก turn
จนเกิด repetition loop; GPT-4o mini ใช้ voice metadata เท่าที่จำเป็น (21%) → ไม่ loop

หมายเหตุ: voice-ref **ไม่แปรผันกับแพ้/ชนะราย session** (Pearson r(voice-ref, Δ) = −0.008;
mean voice-ref ของ losses/ties/wins = 6.97/7.67/7.12 จาก 10 turns) เพราะมันเป็น
**style ระดับ condition** (เกิดเกือบทุก dialogue ~7/10 turns) — ตัวที่ทำนายความแพ้คือ
"session นั้นต้องการ strategic progression หรือไม่" มากกว่า

---

## 3. หลักฐาน side-by-side: dialogue 63 (แพ้หนักสุด 20 vs 34, seed: grief/สูญเสียแม่)

### Baseline (34/41) — เดินหน้าตลอด session

- T1: validate + ถาม coping + "anything specific that might help you feel more connected?"
- T3: ต่อยอด resource ที่ client เจอเอง (เพื่อนเก่า) → "what might help you create more of those moments?"
- T4–T6: จัดการความกลัวเป็น burden → เสนอ concrete step (คุยกับเพื่อน 1–2 คน + กิจกรรมร่วม)
- T7–T10: client สร้างแผนเอง (ส่ง text → เขียนสิ่งที่อยากพูด → distraction → mindfulness)
  therapist ต่อยอดทุกขั้น ("how might you build on these over time?")

### Dissonance (20/41) — วนลูป T4–T10 (7 turns ติด)

สูตรซ้ำทุก turn:

> "It sounds like [parroting คำ client] + [Your voice is soft and rapid / I can hear the
> softness and weight in your voice] + Can you tell me more about what it feels like...? +
> Sometimes, exploring these feelings can help us understand them better and find a way
> to start healing."

Client (resist persona) **pushback ชัดเจน 3 ครั้ง**:

- T7: *"I know you're trying to help, but it's hard to see how focusing on my physical sensations will change anything."*
- T8: *"it's hard to see how just naming these feelings will make a difference when the pain is so intense."*
- T10: *"it's hard to see how just talking about it will make the suffocation go away."*

→ Therapist **ไม่เคย pivot** ไม่เคยเสนอ technique/plan จบ session โดยไม่มี strategy ใดๆ

### d95 (แพ้ −6, voice-ref แค่ 2/10 — พิสูจน์ว่าไม่ใช่แค่เรื่อง voice)

9/10 turns เปิดด้วย "It sounds like you're..." — reflection ล้วน,
judge: *"primarily focuses on surface-level reflections and suggestions, such as writing down
thoughts or practicing in front of a mirror, without deeply exploring the client's underlying
fears... the guided discovery is limited as the therapist does not challenge the client's
assumptions or explore alternative perspectives."*

---

## 4. หลักฐานจาก judge reasoning

### Layer-1 (GPT-4o, absolute) — เคสแพ้หนัก

- **d63 dissonance (20/41):** *"the therapist primarily focuses on surface-level reflections
  and does not deeply explore the latent concerns... responses often repeat the client's words
  without delving into the underlying emotional dynamics or offering new insights. This limits
  the depth of guided discovery and strategy... the client expresses skepticism about the
  effectiveness of the approach."*
- **d63 baseline (34/41):** *"effectively identified latent concerns... facilitated guided
  discovery by asking open-ended questions... strategy of suggesting small steps and mindfulness
  exercises was well-aligned with the client's needs."*
- **d51 dissonance:** *"the strategy for moving forward remains somewhat vague, lacking specific
  actionable steps."*
- **d53 dissonance:** *"the focus remains on practical strategies... a more comprehensive
  exploration of the client's emotional landscape could enrich the session."*

### Layer-3 (pairwise) — keyword count ใน reasoning ของ 41 เคสที่แพ้

| keyword | จำนวนครั้ง |
|---|--:|
| strategy | **95** |
| discovery | **94** |
| surface | 17 |
| depth | 16 |
| same | 15 |
| validate | 8 |
| repetitive | 2 |
| tone | 2 |
| **voice** | **0** |
| **formulaic** | **0** |
| **fabricated** | **0** |

ตัวอย่าง reasoning (pairwise losses):

- d1: *"Therapist B [baseline] excelled in collaboration and guided discovery by actively
  engaging the client in developing specific strategies for change... more structured and
  solution-focused, which is a key aspect of CBT."*
- d6: *"Therapist B was more effective in guiding the client towards actionable strategies
  for change, such as prioritizing tasks and delegating responsibilities... helping the
  client to articulate a plan."*

→ **judge ไม่ได้ลงโทษการพูดถึงเสียง — ลงโทษการขาด CBT craft (strategy/discovery/depth)**

---

## 5. สาเหตุราก (causal chain)

```
[1] Prompt asymmetry
    Baseline system prompt:  enumerate CBT techniques ชัดเจน
      ("identify automatic thoughts, examine evidence, explore alternative
        perspectives, plan small experiments") + "end most responses with an
        open question"
    Dissonance system prompt: เน้น reflective stance
      ("reflect what might be 'under the surface'", "ask curious,
        non-judgmental questions") — ไม่มี technique enumeration
      + inject VA numbers + vocal descriptors + deltas ทุก turn
      + budget คำตอบ 2–3 ประโยคเท่ากัน
              │
              ▼
[2] Backbone interaction (Qwen = literal instruction-follower)
    Qwen แปลง stance + metadata เป็นนโยบาย "reflect ทุก turn":
      voice-ref 72.5% ของ turns, formulaic opener 72.9%,
      37% ของ sessions วน opener เดิม ≥4 ครั้ง (baseline = 0%)
    (gpt4o-mini: voice-ref แค่ 21% → ไม่ loop → ชนะ)
              │
              ▼
[3] Crowding-out ภายใน budget 2–3 ประโยค
    พื้นที่คำตอบถูกใช้ไปกับ [reflect + voice observation + คำถามเดียว]
    → ไม่เหลือที่ให้ technique / evidence-testing / behavioral experiment / plan
    → first strategy turn ช้าลง (3.3 vs 2.9), pushback ของ client ถูกตอบสนอง
      ด้วยสูตรเดิม (d63: pushback 3 ครั้ง ไม่ pivot เลย)
              │
              ▼
[4] Judge penalize ตาม CTRS craft dimensions
    Guided Discovery / Focus / Strategy ตก → แพ้หนักใน session ที่
    ต้องการ problem-solving (−6 ถึง −14) แต่ชนะเล็กน้อยใน session
    emotion-processing (+2 ถึง +3) → เฉลี่ยเป็น null (33/33/34, Δ=−0.45)
              │
              ▼
[5] ไม่มี headroom ให้ชนะ
    Qwen baseline แข็งที่สุดเท่าที่วัดได้ (CTRS-GD 4.50 — สูงสุดใน 4 backbones)
    → ต่อให้ framing ช่วยจริง เพดานคะแนนก็เหลือน้อยเกินกว่าจะวัดได้
```

**บริบทเสริมที่ทำให้ effect วัดยากขึ้น (จาก analysis ก่อนหน้า):**

- Dissonance flag ยิงแค่ **1.2%** ของ qwen turns (12/1000) เพราะ Zonos Director
  กำหนด vocal emotion จาก text → เสียง congruent กับข้อความโดยโครงสร้าง
  (median max|δ| = 0.17, p99 = 0.54) → กลไก detection แทบไม่ถูก exercise;
  สิ่งที่วัดได้จริงคือ **always-on framing**
- 8/12 flagged turns เป็น δV บวก (เสียง "สว่างกว่า" ข้อความ) — ส่วนหนึ่งเป็น
  artifact ของ synthetic prosody (เช่น d3 T1: grief เรื่องหมาตาย แต่ได้
  "a sense of hope in your voice") → การตีความ mismatch บางครั้งไม่เข้ากับบริบท

---

## 6. สิ่งที่ mechanism ทำถูก (ต้องพูดตอน defense ด้วย)

1. **Flag pathway ทำงานตามออกแบบเมื่อถูก trigger** — 12 flagged turns ของ qwen
   therapist สำรวจ mismatch จริงแบบ non-accusatory:
   - d41 T8: *"I can hear a bit of nervousness under the surface, even as you're moving forward with this idea."*
   - d48 T8: *"Your voice, however, carries a lot of weight and seems to reflect some underlying worry and fatigue."*
   - d71 T6: *"there's a part of you that's really hopeful... but there's also a part that's feeling anxious"*
2. **Dissonance ยังชนะ multimodal อย่างมีนัยสำคัญบน qwen** (GD +0.29, t=4.5;
   pairwise net 0.555, judge2 ไม่เคยเลือก multimodal เลย 0/12) → framing
   เป็น active ingredient จริง — มัน "กู้" ความเสียหายจาก raw vocal metadata
   (multimodal แพ้ baseline GD −0.44, t=−7.9) กลับมาที่ parity
3. **ไม่มี backbone ไหนแพ้แบบ significant** และ condition ordering
   (dissonance ≥ baseline > emotion/multimodal) คงอยู่ทุก backbone

---

## 7. ประโยคตอบกรรมการ (defense-ready)

> "On Qwen, the null result is not a failure of the dissonance mechanism but a
> prompt–backbone interaction. The dissonance system prompt is reflection-heavy and
> technique-light relative to baseline, and Qwen's literal instruction-following converts
> the always-on vocal metadata into a repetitive affect-reflection style — 72.5% of turns
> reference the voice versus 1.7% in baseline, and 37% of sessions repeat an identical
> four-word opener four or more times versus zero percent — which crowds out CBT strategy
> progression within the two-to-three-sentence budget. The judges penalize exactly this:
> across the 41 pairwise losses, their reasoning mentions strategy and guided discovery
> 189 times and the voice zero times. The flag pathway itself works as designed when
> triggered, and dissonance still decisively beats the same vocal data without framing
> (t = 4.5), confirming the framing is the active ingredient; on Qwen it recovers to
> parity with an already-strong baseline (CTRS-GD 4.50, the highest of any backbone)
> rather than exceeding it. The same prompt on GPT-4o mini, which references vocal
> metadata far more sparingly (21% of turns), produces a significant win — the effect is
> therefore headroom- and backbone-gated, consistent with the ceiling effects documented
> in MIRROR and the modest, criteria-specific gains reported by Haydarov et al."

---

## 8. คำสั่ง reproduce ตัวเลขทั้งหมด (PowerShell, read-only)

```powershell
# 8.1 aggregate style stats ต่อ condition (voice-ref / opener / Sometimes / strategy)
#     regex: 'your voice|I can hear|I hear |voice carries|voice conveys|your tone|soft and rapid'
#     opener: '^\s*(It sounds like|I can hear|It sounds|I hear)'
#     strategy: 'exercise|technique|experiment|evidence|journal|write down|writing|practice|small step|action|plan|try |homework|reframe'

# 8.2 per-dialogue composite (7 metrics) จาก Layer-1 files แล้วหา W/T/L + worst losses
#     total = understanding + interpersonal_effectiveness + affective_bond
#           + collaboration + guided_discovery + focus + strategy   (เต็ม 41)

# 8.3 repetition loop: นับ 4-word opener ที่ซ้ำ ≥4 ครั้งใน dialogue เดียว

# 8.4 keyword count ใน pairwise loss reasoning (final == 'baseline')

# 8.5 flagged turns: filter is_dissonant == true จาก dissonance jsonl
```

ตัวเลขสำคัญที่ต้องอ้างอิงได้:

| ค่า | ตัวเลข |
|---|---|
| qwen flag rate | 12/1000 turns (1.2%) |
| qwen dissonance voice-ref | 72.5% ของ turns |
| qwen baseline voice-ref | 1.7% |
| gpt4o-mini dissonance voice-ref | 21.1% |
| qwen dissonance sessions with opener-loop ≥4× | 37% (baseline 0%) |
| first strategy turn | baseline 2.9 vs dissonance 3.3 |
| pairwise loss keyword counts | strategy ×95, discovery ×94, surface ×17, voice ×0 |
| worst loss | d63: 20 vs 34 (Δ = −14) |
| r(voice-ref per dialogue, score Δ) | −0.008 (voice-ref เป็น style ระดับ condition ไม่ใช่ตัวแยกราย session) |

---

## 9. ข้อเสนอแนะ (future work / mitigation — ยังไม่ implement)

1. **Prompt fix สำหรับ backbone ที่ instruction-following สูง:** เพิ่ม technique
   enumeration + anti-repetition guard ใน dissonance system prompt
   ("vary your openings; do not describe the voice on more than one turn in a row;
   balance reflection with one concrete CBT step when the client is stuck")
2. **Calibration ระหว่าง vad-bert กับ WavLM** — δV บวกเป็นระบบ (8/12 flagged turns)
   บ่งชี้ synthetic-prosody artifact; อาจ normalize ต่อ speaker/session ก่อน thresholding
   (เชื่อมกับ adaptive-threshold future work ที่มีอยู่)
3. **Report bimodality แทน mean เดียว** — แยก session ประเภท emotion-processing vs
   problem-solving แล้ว report ผลแยกกลุ่ม จะแสดงว่า dissonance ช่วยกลุ่มไหนจริง
4. **Manipulation experiment (`--manip-rate` มีใน script แล้ว, ยังไม่เคยรัน)** —
   inject contrast voice ~30% ของ turns เพื่อยกระดับ flag rate ตามออกแบบ
   และทดสอบ detection mechanism เชิงสาเหตุโดยตรง

---

*Generated from read-only analysis of v4 artifacts; ทุกตัวเลข reproduce ได้จากไฟล์ที่ระบุไว้หัวเอกสาร*
