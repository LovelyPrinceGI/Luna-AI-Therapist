# Defense Q&A — คำถามที่กรรมการน่าจะถาม + คำตอบที่มีหลักฐานรองรับ

เตรียมจาก: cause_qwen.md, ai_multibackbone_report_v4.md, การ audit เทียบ paper ต้นทาง (CACTUS/MIRROR/Haydarov/SpeechCueLLM) และตัวเล่ม thesis (หลังแก้ A1/A2 แล้ว)

---

## Q1. ทำไม Qwen Dissonance-Aware ไม่ชนะ Baseline ทั้งที่เป็น condition หลัก?

**Short answer:** ไม่ใช่ความล้มเหลวของ dissonance signal แต่เป็น **prompt × backbone interaction** ที่เราวิเคราะห์เจอกลไกครบแล้ว (Section 6.1.3 ของเล่ม)

**Evidence chain:**
- ผลเป็น "null with catastrophic tail": ชนะสูงสุด +3 แต่แพ้หนักสุด −14 (d63), −8, −7, −6 → ค่าเฉลี่ย −0.45, W/T/L 33/33/34
- Style audit (1,000 turns/condition): Qwen dissonance พูดถึง voice **72.5%** ของ turns (baseline 1.7%), เปิด turn ด้วย "It sounds like/I can hear" **72.9%** (baseline 8.0%), **37%** ของ sessions วน opener เดิม ≥4 ครั้ง (baseline **0%**)
- Prompt เดียวกันบน gpt4o-mini พูดถึง voice แค่ **21.1%** → style นี้เป็นพฤติกรรมเฉพาะ backbone (Qwen = literal instruction-follower) ไม่ใช่ผลจาก prompt ล้วนๆ
- Judge reasoning ใน 41 เคสที่แพ้: strategy ×95, discovery ×94, surface ×17 — **voice ×0** → judge ลงโทษ "session ไม่ advance" ไม่ใช่ "พูดถึงเสียง"
- d63: client pushback 3 ครั้ง ("it's hard to see how just talking about it will make the suffocation go away") แต่ therapist วน reflection + voice-narration เดิมทั้ง 7 turns สุดท้าย — ไม่ pivot เลย
- Baseline prompt มี technique enumeration ("identify automatic thoughts, examine evidence, ... plan small experiments") แต่ dissonance prompt เน้น reflective stance → ใน budget 2–3 ประโยค reflection เบียด strategy ออก

**Closing line:** "On Qwen the framing recovers the damage raw vocal metadata causes (beats Multimodal, t = 4.5) and restores parity with the strongest text-only baseline in the study (Guided Discovery 4.50) — a headroom effect, not a mechanism failure."

---

## Q2. ทำไมไม่ re-run Qwen หรือเปลี่ยน backbone ที่ผลดีกว่า?

**Short answer:** เพราะ protocol **pre-registered** ก่อนรัน (interpretation criteria固定ไว้ใน Ch.4) และการเลือก backbone/seed ใหม่หลังเห็นผลคือ **selection on outcome** ซึ่งทำลายความน่าเชื่อถือของทั้ง 1,600 sessions

**Talking points:**
- เราประกาศ qwen เป็น "honest neutral case" ใน results ตั้งแต่แรก (F4) — การ report ตามจริงคือจุดแข็งทางระเบียบวิธี
- Failure mode เป็น systematic (style stats ต่างกัน 72.5% vs 1.7% อย่างเสถียร) ไม่ใช่ sampling noise → re-run จาก distribution เดิมให้ผลแบบเดิม (Δ = −0.45, t = −1.9 → โอกาสพลิกเป็นบวก ~3%)
- เราทำสิ่งที่ถูกต้องแทน: **post-hoc error analysis บน artifacts เดิม** (ไม่มีการ generate ข้อมูลใหม่) เพื่ออธิบายกลไก
- ไม่มีเซลล์ใดแพ้แบบ significant — claim ของ thesis (headroom-gated benefit + ordering คงอยู่ทุก backbone) สอดคล้องกับข้อมูลครบทั้ง 4 backbones

---

## Q3. Flag ยิงแค่ ~1% ของ turns — แปลว่ายังไม่ได้ทดสอบ dissonance detection จริงๆ?

**Short answer:** ยอมรับตรงๆ ว่า detection mechanism ถูก exercise น้อยในการทดลองหลัก (เพราะ TTS director สร้างเสียงจาก text → congruent โดยโครงสร้าง) แต่ (1) สิ่งที่เราวัดได้และ claim คือ **always-on framing** ซึ่งชนะ multimodal อย่างมีนัยสำคัญทุกที่ที่วัดได้ (2) **flag pathway ถูก verify end-to-end** ด้วยเสียงมนุษย์จริงใน demo (δV = −1.05 → targeted exploration, คะแนน 34 vs 27) และ (3) controlled manipulation experiment (`--manip-rate` implement แล้ว) ถูกระบุเป็น future work ชัดเจน

**ตัวเลข:** flag rate 0.8% (gpt4o-mini), 1.2% (qwen), 0.9% (deepseek), 3.7% (claude); median max|δ| ≈ 0.16–0.22 — รายงานใน Ch.5 (Table flagrate) + Ch.6 limitations แล้ว

**ถ้าถูกถามว่า "แล้ว dissonance vs multimodal ที่ชนะล่ะ คืออะไร":** คือผลของ **interpretive frame** — deltas + flag + guidelines ทุก turn ทำให้โมเดลรู้ว่า "เสียงตรงกับข้อความ (delta ≈ 0) → เชื่อข้อความแล้วขุดลึก" แทนที่จะ narrate ตัวเลขดิบแบบ multimodal (ที่แพ้ baseline t = −7.1/−7.9) — นี่เองคือ finding หลักของ thesis ("the dissonance computation, not the vocal data itself, is the active ingredient")

---

## Q4. Haydarov คำนวณ face↔speech ไม่ใช่ text↔speech — การ adapt นี้ faithful ไหม?

**Short answer:** เรา adopt **threshold rule เดิม** (max-form, T = 0.5) และ **substitute channel**: เปลี่ยน FER (หน้า) เป็น text-emotion channel เพราะ deployment目标是 text-based teletherapy (กล้องปิด) — ชัดเจนใน Ch.2 (แก้แล้ว), Ch.3 Phase 4 (มีสูตร SER/FER ต้นฉบับ + เหตุผล), และ Ch.6 Table past-work

**Talking points:**
- Haydarov เองอธิบาย dissonance ระดับ concept ว่า "divergence between the affective tone embedded in the verbal content and that conveyed through non-verbal modalities" — การจับคู่ text (verbal) ↔ speech (non-verbal) สอดคล้องกับ framing ระดับนี้
- เรายัง extend binary tag ของเขา ("Patient (emotional dissonance):") เป็น continuous deltas + flag + interpretive guidelines ซึ่ง rich กว่า
- Boundary ≥ vs > ต่างกันเฉพาะ |δ| = 0.5 เป๊ะ — ระบุ footnote ไว้ใน Ch.3 แล้ว
- ผลลัพธ์เชิงคุณภาพ converge กับเขา: dissonance-guided prompting ดีกว่า base — แต่เราเพิ่ม boundary conditions (headroom-gated) ที่เขาไม่ได้วัด (เขาใช้ single-turn intervention, เราใช้ 10-turn sessions × 4 backbones)

---

## Q5. MIRROR ตัด Strategy ออกเพราะ length bias — ทำไมคุณรวม?

**Short answer:** Strategy เป็น 1 ใน 6 items ของ CTRS/COUNSELINGEVAL (CACTUS ใช้ครบ 6) — เราเลือกคง item set เต็มของ CTRS และจัดการ length bias เชิงเครื่องมือแทนการตัด item:
1. Pairwise prompt สั่งชัด "Do NOT reward longer responses"
2. Post-hoc length audit ทุกเซลล์: |r(Δlength, outcome)| ≤ 0.17
3. ชัยชนะที่ใหญ่ที่สุด (net 0.800) เกิดกับ transcripts ที่ **สั้นกว่า** เฉลี่ย 1,310 chars → verbosity อธิบายผลไม่ได้

ระบุไว้ใน Appendix A (note ใหม่) + Ch.4 แล้ว

---

## Q6. "Emotional Subtext" / "Latent Concerns" มาจาก MIRROR ใช่ไหม?

**Short answer:** **ไม่ใช่** — เป็น operational constructs ของเราเองที่เขียนไว้ใน judge prompt เพื่อเล็งความสนใจของ evaluator ไปที่ say-vs-feel gap; สิ่งที่มาจาก MIRROR/COUNSELINGEVAL คือ **dimensions + scales** (skills 0–6, WAI bond 1–5) — ตอนนี้ระบุชัดทั้งใน Appendix A และ Ch.4 แล้ว (concept ใกล้เคียงใน MIRROR: "deeper emotions", "Is something else on your mind?")

---

## Q7. κ ระหว่าง judges = 0.217 (fair) — เชื่อผลได้แค่ไหน?

**Talking points:**
- κ ตกเพราะ **tie-vs-commit style** ระหว่าง model families (GPT-4o judge default tie เมื่อ swap-orders ไม่ตรงกัน; Claude-judge commit บ่อยกว่า) + เซลล์ claude ที่ sessions แยกไม่ออกจริง (99–100% ties)
- **Directional agreement 6/6** ในเซลล์ที่ทั้งสอง judges commit — ไม่มีเซลล์ใด lean สวนกัน
- Swap-consistency rule แบบ conservative (ชนะต้องชนะทั้งสอง order) → residual bias ทำได้แค่เพิ่ม tie ไม่มีวันสร้าง win เทียม
- Human-expert validation ของ pairwise subset ถูกระบุเป็น future work อันดับ 1 (ตามแบบ MIRROR ที่ใช้ 200 cases)

---

## Q8. Multimodal แพ้ Baseline — ขัดกับ literature (SpeechCueLLM/Beyond Text) ที่ว่า vocal cues ช่วย?

**Short answer:** ไม่ขัด — เป็น **boundary finding ใหม่** ที่เราเป็นรายแรกที่ document ใน generative therapeutic dialogue: SpeechCueLLM/Sun et al. แสดงผลใน classification/decision tasks; เราพบว่าในการ *generate* คำตอบบำบัด raw vocal metadata โดยไม่มี interpretive frame ทำให้ backbone อ่อนผลิต voice-narration ซ้ำๆ ที่ judge ลงโทษ (qwen multimodal: voice-ref 89.9%, GD −0.44, t = −7.9) — Ch.6 เขียนไว้แล้วว่า "sharpens rather than contradicts the prior work: the value of vocal cues depends on giving the model a reasoning frame"

---

## Q9. Claude: multimodal นอมินัลสูงสุด (5.24) — แย้ง claim ไหม?

**Short answer:** เซลล์ Claude **inconclusive at ceiling** โดย instrument เอง: pairwise 99–100% ties + positional consistency ≈ 0.01 (verdict พลิกตาม order = แยกไม่ออก), absolute scores ต่างกัน 0.15 ภายใน band ที่ judge ให้คะแนน identical; และ second judge (Claude) ที่ commit จริงบนเซลล์เดียวกันเลือก Dissonance เหนือ Multimodal **10 ต่อ 2** — เราเขียนไว้ชัดว่า "read as inconclusive at ceiling, consistent with MIRROR's headroom finding" ไม่ใช่ evidence against

---

## Q10. System prompt ต่างกันระหว่าง conditions — เป็น confound ไหม?

**Short answer:** ต่างแบบ **coordinated by design**: ทุก condition ใช้ identity เดียวกัน ("Luna"), budget เดียวกัน (2–3 sentences), ข้อห้ามเดียวกัน (ห้ามพูดตัวเลข) — ส่วนที่ต่างคือแต่ละ system prompt **อธิบาย fields ที่ condition นั้นได้รับ** (ไม่มี → text VA → +speech VA/descriptors → +deltas/flag/guidelines) ซึ่งแยกออกจาก user-template metadata ไม่ได้โดยธรรมชาติ (prompt จะอ้าง fields ที่ไม่มีไม่ได้) — เล่มแก้ข้อความให้ตรงข้อเท็จจริงนี้แล้ว (Ch.3 Phase 5, Ch.4, Appendix C)
- สิ่งที่ control จริงคือ **information ladder**: baseline ⊂ emotion ⊂ multimodal ⊂ dissonance และ client/director prompts identical ทุก condition
- ข้อค้นพบสำคัญ (framing > raw data) มาจากคู่ multimodal vs dissonance ซึ่ง field sets ซ้อนกันเกือบหมด — ต่างกันที่ deltas + flag + interpretive guidelines เท่านั้น

---

## Q11. Demo session (Appendix B) ใช้เสียงจริง — ทำไม main experiment ไม่ใช้?

**Short answer:** privacy/ethics + scale (1,600 sessions ต้องการ audio ปริมาณมาก) + controllability; TTS ทำให้ exact control over intended vocal emotion แลกกับ fidelity (prosody แคบ → flag rate ต่ำ) — ระบุใน limitations แล้ว; demo เสียงจริงทำหน้าที่ verify **flag pathway** ที่ main experiment exercise ไม่ถึง; future work: real-voice replay + manipulation experiment

---

## Q12. คะแนนเต็มของ judge rubric คือ多少?

**41** = Understanding(6) + Interpersonal Effectiveness(6) + Affective Bond(5) + Collaboration(6) + Guided Discovery(6) + Focus(6) + Strategy(6) — ตัวเล่ม (Appendix B + Ch.4) แก้จาก 45 → 41 แล้ว; คะแนน demo 27 กับ 34 ถูกต้องตาม scores_custom.json

---

## Cheat sheet: ตัวเลขที่ต้องจำ

| ตัวเลข | ค่า |
|---|---|
| Sessions ทั้งหมด | 1,600 (4 backbones × 4 conditions × 100) |
| Pairwise comparisons | 800 × 2 orders = 1,600 judge calls |
| gpt4o-mini: dissonance vs baseline | GD +0.24 (t=3.1), F +0.34 (t=4.7), S +0.17 (t=2.7); pairwise net 0.680, p=0.0001 |
| gpt4o-mini: dissonance vs multimodal | GD +0.75 (t=10.1); net 0.800, p<0.00001 (transcripts สั้นกว่า 1,310 chars) |
| deepseek: pairwise vs baseline | net 0.660, p=0.00004 (absolute saturate ที่ 5.00) |
| qwen: vs baseline | Δ=−0.45, t=−1.90 n.s.; 36W/41L, p=0.649 → **honest null** |
| qwen: dissonance vs multimodal | GD +0.29 (t=4.5); judge2 เลือก multimodal 0/12 |
| claude | 99–100% ties, positional consistency ≈0.01 → ceiling |
| flag rates | 0.8/1.2/0.9/3.7% (gpt4o-mini/qwen/deepseek/claude) |
| qwen style audit | voice-ref 72.5% vs 1.7%; opener-loop 37% vs 0% |
| judge κ | 0.217 (item-level) / directional 6/6 |
| length bias audit | \|r\| ≤ 0.17 ทุกเซลล์ |
| safety | accusatory phrasing 5/4,800 dialogues (~0.01%), กระจายเท่ากัน |
| demo (เสียงจริง) | δV=−1.05 → flag; judge 34/41 vs 27/41 |

---

*เอกสารนี้ใช้เตรียมคำตอบปากเปล่า — ตัวเลขทุกตัว reproduce ได้จาก artifacts ใน repo (ดู cause_qwen.md §8 สำหรับคำสั่ง)*
