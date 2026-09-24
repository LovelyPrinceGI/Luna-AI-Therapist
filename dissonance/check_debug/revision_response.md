# Revision Response Memo (post-final-defense)

Map คอมเมนต์กรรมการ → จุดที่แก้/เพิ่มในเล่มและเอกสารประกอบ (อัปเดตล่าสุด 2026-09-18)

---

## คอมเมนต์ 1: "ใส่ control variables แบบ per-turn record เข้าไปใน report ให้ experiment/methodology ละเอียดขึ้น"

**ตอบ/แก้ที่:**
1. `04-experiment.tex` — subsection ใหม่ `\subsection{Experimental Variables and Controls}` (\label{sec:variables-controls}) พร้อม:
   - `tab:variables`: แถว Manipulated (IV = metadata ladder 4 ระดับ) / Measured (DV = Layer 1-3 + secondary) / Held constant (controls: temp 0.7 ไม่มี vendor sampling additions, shared 100 seeds, persona/director prompts เดียวกัน, budget 2-3 ประโยค + no-numbers rule, τ=0.5, extraction stack, Zonos config, 10-turn stateful scheme, judge protocol) / Recorded per turn
   - `tab:record-fields`: field dictionary ครบ 17 fields (turn, client, therapist, condition, model, val_t/aro_t, val_s/aro_s, text/speech_emotion, vocal_descriptors, delta_*, is_dissonant, emotion_mismatch, audio_path) พร้อม type + ความหมาย/บทบาท
   - Sanitized sample record (dialogue 1, turn 1, dissonance, gpt4o-mini) ใน verbatim block โดยเปลี่ยน `audio_path` เป็น relative path (`voice/dissonance_gpt4o-mini_1_utterance_1.wav`) ไม่ leak path เครื่อง
2. `dissonance/evaluation/ai_multibackbone_report_v4.md` — section ใหม่ `### 2.5 Controlled and Recorded Variables` (IV/DV/controls/recorded fields สรุปแบบ bullet ชี้กลับไปที่ตารางในเล่ม)

**จุดที่อ้างได้ตอนตอบปากเปล่า:** ทุก session re-derive/audit ได้จาก artifacts เพราะ pipeline เขียน record ครบทุก turn (completeness-verified 16/16 cells, 100/100 files)

---

## คอมเมนต์ 2: "ไปดู applicable study ที่ implement dissonance LLM ใน therapeutic domain จริงๆ"

**ตอบ/แก้ที่:**
1. `02-literature.tex` (section Emotional Dissonance and Multimodal Inconsistency):
   - ย่อหน้าใหม่สรุป 3 งานที่ implement จริง: **E-THER** (verbal-visual incongruence + PCT training ลด performative empathy), **CADD** (text-voice dissonance 3 ทิศทาง Masking/Coping/Congruent + journaling app + TTS domain gap), **ATEI** (acoustic-textual inconsistency → depression detection, severity correlation, ECI theory)
   - เขียน **Research Gap ใหม่ให้แม่นขึ้น**: ไม่มีงานไหน inject continuous text-speech VA delta + threshold flag เป็น prompt-level signal สำหรับ multi-turn therapist generation + isolated causal contribution แบบ controlled multi-backbone (Haydarov = binary cue single-turn; CADD = classifier + journaling; E-THER = trained VLM)
   - เพิ่ม 3 แถวใน `tab:related-work-summary`
2. `06-discussion.tex`:
   - เพิ่ม 3 แถวใน `tab:past-work` (ก่อนแถว Ours)
   - ย่อหน้าใหม่ "Positioning within the dissonance lineage" หลังตาราง
3. `08-references.bib` — เพิ่ม `ether2025` (arXiv:2509.02100), `cadd2026` (arXiv:2604.27517), `atei2024` (arXiv:2412.18614)
4. เอกสารเตรียมตอบ: `check_debug/applicable_studies_summary.md` — สรุปภาษาไทย 4 งาน (รวม Haydarov) + ตารางเทียบเร็ว + คำถามที่อาจโดนถามและวิธีตอบ

---

## คอมเมนต์เดิมที่แก้ไปแล้วในรอบก่อนหน้า (อ้างอิงได้)

| คอมเมนต์/ประเด็น | แก้ที่ |
|---|---|
| Qwen null ต้องอธิบายกลไก | §6.1.3 post-hoc error analysis + Table 6.1 style audit + pointer หลัง Table 5.6 + Ch.4 sampling disclosure + §6.3 trade-off |
| กลัวซ้ำเพราะ infra/quantization/Qwen bug | §6.1.3 rule-out paragraph (degeneration signatures 0/8,010 turns, dose-response, flex opt-in) + citations (qwen25tech, Holtzman, Welleck, Keskar) + cause_qwen.md §10 + defense_qa.md Q13 |
| คะแนนเต็ม rubric ผิด (45) | appendix.tex + 04-experiment.tex → 41 |
| Haydarov ถูกบรรยายผิดเป็น text↔speech | 02-literature.tex + 03-methodology.tex Phase 4 (ระบุ face↔speech ต้นทาง + การ adapt) |
| "Emotional Subtext/Latent Concerns" ไม่ใช่ศัพท์ MIRROR | appendix.tex note + 04-experiment.tex |
| Strategy ทั้งที่ MIRROR ตัดออก | appendix.tex note + 04-experiment.tex (justify + length-bias mitigations) |
| "identical system instructions" ไม่จริง | appendix.tex + 04-experiment.tex + 03-methodology.tex Phase 5 |
| SpeechCueLLM ไม่ได้ cite ตรงจุดใช้ | 02-literature.tex + 03-methodology.tex Phase 2 |
| อ้างผล stateless pilot ที่ไม่ได้เซฟ | ลบทุกจุด (05-results, 06-discussion) + ลบป้าย v1/v4 ทั้งเล่ม |
| ทำไมไม่ใช้ dominance/VAD | 03-methodology.tex Phase 3.5 paragraph + bib russell1977threefactor |

---

*ไฟล์นี้ใช้ยื่นประกอบ revision / ทวนความครบถ้วนก่อนส่ง — ทุกจุดแก้มี backup .bakN ในโฟลเดอร์เดียวกัน*
