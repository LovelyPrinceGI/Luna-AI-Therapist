# สรุป Applicable Studies: งานที่ implement dissonance/incongruence ใน therapeutic domain จริง

อ่านด่วนก่อน revision/ตอบกรรมการ — แต่ละงาน: ทำอะไร / ข้อมูล / ผล / เกี่ยวอะไรกับเรา / จุดที่เขาอาจถาม

---

## 1) Haydarov et al. 2025 (EMNLP main) — มีในเล่มแล้ว
- **ทำอะไร:** นิยาม emotional dissonance เชิงคำนวณ (inspired โดย Hochschild's emotional labor) = mismatch ระหว่าง VA จากเสียง (SER) กับ VA จากหน้า (FER) บนวิดีโอ therapy จริง 1,441 คลิป; flag เมื่อ max(|ΔV|,|ΔA|) > T, T=0.5; พอ flag ก็ prepend cue "emotional dissonance" หน้า utterance คนไข้ใน prompt ของ intervention LLM
- **ผล:** dissonance prompting เพิ่ม empathy rating เทียบ base prompt (Δ ≈ 0.445 สเกล 1–5); แต่ comprehensiveness/applicability ไม่เพิ่ม; และ **LLM judge ไม่ตรงกับผู้เชี่ยวชาญ**
- **ต่างจากเรา:** ช่อง face↔speech (เรา text↔speech), single-turn intervention (เรา multi-turn), cue แบบ binary (เรา continuous deltas + flag + guidelines), ไม่มีการควบคุม backbone
- **ถ้าถูกถาม:** "เขาทำก่อนคุณใช่ไหม?" → ใช่ เขาคือ source ของ threshold rule และแนวคิด dissonance-as-prompt-cue; เรา adapt channel pair เป็น text↔speech (teletherapy กล้องปิด) และขยายเป็น continuous signal + controlled multi-backbone isolation (ระบุชัดใน Ch.2/Ch.3/Ch.6 แล้ว)

## 2) E-THER (Tahir et al. 2025, arXiv:2509.02100)
- **ทำอะไร:** dataset multimodal บทสนทนา therapy จริง annotate **verbal-visual incongruence** ตาม Person-Centered Therapy (PCT) 3 แบบ: Minimizing (สีหน้า distress กว่าคำพูด), Contradiction (สีหน้าตรงข้ามคำพูด), No incongruence + engagement scores; ใช้ train VLM (IDEFICS, VideoLLaVA) ด้วย incongruence-aware learning
- **ผล:** โมเดลที่ train กับ incongruence ชนะ SOTA ในtraits เช่น sustaining therapeutic engagement และ **ลด artificial/exaggerated linguistic patterns** (performative empathy)
- **ต่างจากเรา:** ช่องคำ↔สีหน้า (ไม่ใช่เสียง), เป็น training-based VLM ไม่ใช่ prompt injection, task เป็น empathic response ตาม PCT ไม่ใช่ CBT generation
- **ถ้าถูกถาม:** "ทำไมไม่ train แบบเขา?" → เราเลือก prompt-level เพื่อ architecture-agnostic + cost (H2 testable ทุก backbone); และผลของเขาเสริมเรา: incongruence-awareness ลด performative empathy = ทิศเดียวกับที่ dissonance framing ทำให้ therapist เราลึกขึ้น
- **ประโยคเชื่อม:** เขาชี้ว่า mismatch เป็นสัญญาณวินิจฉัยใน counseling theory (Rogers' congruence) — รองรับ framing ของเราว่า mismatch = hidden affect ไม่ใช่ noise

## 3) CADD / ReflectJournal (Lee 2026, arXiv:2604.27517) — ใกล้เคียงเราสุด
- **ทำอะไร:** formalize **text↔voice dissonance** (ช่องเดียวกับเรา!) เป็น directional 3-class: **Masking** (text บวก + เสียงลบ = suppression), **Coping** (text ลบ + เสียงบวก = reappraisal), **Congruent**; สร้าง CADD-Journal (1,800 ตัวอย่าง TTS, shared-sentence-pool: ประโยคเดียวกัน render หลายเสียง → บังคับโมเดลใช้ acoustic จริง); โมเดล DACM (dual-encoder + asymmetric cross-modal attention) macro-F1 0.711; 앱 ReflectJournal ให้ reflective prompt ตอน detect mismatch เช่น *"Your words sound positive, but your voice seems a bit heavy. Is there anything else on your mind?"*
- **ผลสำคัญ:** zero-shot ไป corpora เสียงจริง (CMU-MOSEI, IEMOCAP) **ร่วงใกล้ random** → TTS→real domain gap ใหญ่มาก
- **ต่างจากเรา:** เป็น classifier + journaling/self-reflection app ไม่ใช่ therapist dialogue generation; ไม่ใช้ VA delta ต่อเนื่อง/threshold ใน prompt; ไม่มีการเปรียบเทียบ conditions/backbones
- **ถ้าถูกถาม:** "งานนี้ทำก่อนคุณ ทำไมclaim first?" → เขาทำ **detection** สำหรับ self-reflection; เราเป็นงานแรกที่ใช้ text-voice VA delta เป็น **prompt-level signal สำหรับ multi-turn CBT therapist generation** + isolated causal contribution (เขียนชัดใน Research Gap Ch.2 + positioning Ch.6)
- **ใช้เสริมเรา:** (ก) directional classes ของเขา = การอ่านเครื่องหมาย δ ของเรา (δV ลบ = Masking-like, δV บวก = Coping-like) — อ้างเป็น future work ได้; (ข) domain gap TTS→real ของเขา **ยืนยัน limitation เรา** เรื่อง synthetic speech และ flag rate ต่ำ

## 4) ATEI (Su et al. 2024, arXiv:2412.18614)
- **ทำอะไร:** ดึง **Acoustic-Textual Emotional Inconsistency (ATEI)** จาก counseling conversations ด้วย multimodal cross-attention (acoustic vs textual sentiment ไม่ตรงกัน = inconsistent) แล้ว fuse เข้า depression detection model + learnable scaling ตาม severity
- **ผล:** เพิ่ม subject-level accuracy 0.73–5.51% เทียบ baseline; และดีกรี inconsistency **สัมพันธ์กับ depression severity** (grounded ใน Emotion Context-Insensitivity theory: คนซึมเศร้าเล่าเรื่องลบด้วยเสียงสงบนิ่งผิดปกติ = masked affect พอดี)
- **ต่างจากเรา:** detection/diagnosis ไม่ใช่ generation; ใช้ sentiment-label mismatch ไม่ใช่ VA delta
- **ถ้าถูกถาม:** "มีหลักฐานคลินิกไหมว่า mismatch มีความหมาย?" → มี: ATEI แสดงว่า inconsistency correlate กับ severity; E-THER อ้าง counseling theory ว่า verbal-visual incongruence diagnostically meaningful — รองรับ interpretation ของเราว่า dissonance = hidden affect signal ไม่ใช่ artifact

---

## ตารางเทียบเร็ว (จำไว้ตอบ)

| งาน | ช่อง | กลไก | Task | เราต่างยังไง |
|---|---|---|---|---|
| Haydarov 2025 | face↔voice | threshold + binary cue ใน prompt | single-turn intervention | เรา text↔voice, multi-turn, continuous deltas, controlled backbones |
| E-THER 2025 | word↔face | incongruence-aware training | empathic VLM (PCT) | เรา prompt-level, CBT, ไม่ train |
| CADD 2026 | word↔voice | classifier 3 ทิศทาง + app prompt | reflective journaling | เรา therapist generation + causal isolation |
| ATEI 2024 | word↔voice | inconsistency features + fusion | depression detection | เรา generation ไม่ใช่ detection |

**One-liner ตำแหน่งของเรา:** "First to inject text-voice VA dissonance as a prompt-level therapeutic signal for multi-turn CBT response generation, with its causal contribution isolated against identical vocal inputs across four backbones."
