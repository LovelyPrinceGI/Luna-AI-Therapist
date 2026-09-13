# Defense Q&A Sheet — Luna AI Therapist (v4 results)

ชีทซ้อมตอบกรรมการสอบ final defense สำหรับบทผลการทดลอง (latex/05-results.tex)
หลักการตอบทุกข้อ: **ยอมรับ boundary condition อย่างซื่อสัตย์ + โชว์หลักฐานหลายชั้นที่ลงตัวกัน + ปิดด้วย future work** — ห้าม overclaim เกินกว่าที่ตารางรองรับ

แหล่งอ้างอิง: `05-results.tex` (บท RESULT), `dissonance/evaluation/ai_multibackbone_report_v4.md` (report F1–F6), `dissonance/evaluation/evaluation_outputs/pairwise_summary.md`

---

## ตารางคำถาม–คำตอบ

| # | กรรมการอาจถาม | คำตอบ / ข้อมูลรองรับ | ที่มา |
|---|---|---|---|
| 1 | "ทำไม qwen ไม่ดีขึ้น?" | รายงานตรง ๆ เป็น **boundary condition**: effect เป็น headroom-gated — Baseline ของ qwen แข็งแรงอยู่แล้ว (CTRS-G 4.50 สูงสุดในงาน) และ **สามชั้นลงตัวยันกันว่า neutral จริง ไม่ใช่ noise**: paired t = −1.90 n.s., pairwise 36/41 p = 0.649, judge ทั้งคู่ lean Baseline (7–4 และ 6–2) แต่ dissonance ยังชนะ Multimodal (Layer 2 t = 2.53) และ Emotion (t = −3.19) → สิ่งที่ fail คือ "ชนะ baseline ทุกตัว" ไม่ใช่กลไก; adaptive threshold ที่ปรับตาม context = future work | Tab 5.2/5.5/5.6/5.8; report F4 |
| 2 | "claude แยกรไม่ออกเลย?" / "ทำไมตาราง claude ให้ Multimodal ชนะ?" | **Ceiling effect**: ทุก condition > 5.0 ทั้งหก metric; pairwise tie 99–100% พร้อม position consistency ≈ 0 = คำตัดสินพลิกเมื่อสลับลำดับ ⇒ เซสชันเทียบเท่ากันที่ความละเอียดของ judge ไม่ใช่ "ไม่มีผล"; MIRROR ก็พบ headroom แบบเดียวกันในโมเดลแข็งแรง | Tab 5.4/5.7; report F3; MIRROR |
| 3 | (ดักหมัดแย้งข้อ 2) "แต่ paired t ของคุณเอง = −3.47 บอก dissonance แพ้ multimodal บน claude?" | ผลต่าง nominal −0.15/metric บน **scale ที่อิ่มตัว** (judge ให้คะแนน identically เกือบทุกเซสชัน) จึง sig ทางสถิติแต่ **ใต้ discriminability ของ pairwise** (tie 100/100, PosC 0.00) และ judge ข้ามครอบครัว (Claude) ที่ commit บน transcript เดียวกันเลือก Dissonance 10–2 ⇒ อ่านเป็น ceiling artifact ไม่ใช่ความแพ้ | Tab 5.3/5.7/5.8; §5.2.4 |
| 4 | "κ แค่ 0.217 เชื่อ judge ได้ไง?" | κ ต่ำเพราะ **tie-vs-commit style** เป็นหลัก: กฎ swap-consistency ทำให้ judge1 tie เกือบหมดบนเนื้อที่แยกไม่ออก (Claude cells tie 26/26) ขณะที่ judge2 commit บน transcript เดียวกัน (และกลับกันที่เซลล์ gpt4o-mini); ในเซลล์ที่มี signal จริง κ = 0.50–0.60 (moderate–good); ระดับทิศทาง: ใน 6 เซลล์ที่ judge1 มี lean judge2 ไม่เคย lean ตรงข้าม และตรงกันทุกเซลล์ใน 5 เซลล์ที่ทั้งคู่ lean; การออกแบบ 3 ชั้น = ไม่มี instrument เดียวแบกข้อสรุป | Tab 5.8; §5.4 |
| 5 | "flag dissonance ยิงแค่ ~1% แล้วจะมีผลได้ไง?" | ผลวัดมาจาก **always-on framing**: prompt ให้ δV, δA ต่อเนื่องทุก turn พร้อมคำแนะนำตีความ ("large |delta| = น้ำเสียงกับคำพูดคนละทิศ ให้สำรวจอย่างอ่อนโยน") ไม่ใช่จาก flag binary; หลักยัน: flag rate ไม่ track ขนาด gap (Claude flag สูงสุด 3.7% แต่ ceiling; gpt4o-mini flag ต่ำสุด 0.8% แต่ gap ใหญ่สุด); เคส flag จริง (δV = −1.05) ในเดโมเสียงจริงทำงานตามออกแบบ | §5.1.3 (Tab 5.3); demo §chat-demo |
| 6 | "ทำไม Multimodal แย่กว่า Baseline? pipeline เสียงพังเหรอ?" | นี่คือ **mechanistic finding ของงาน**: vocal metadata ดิบโดยไม่มีการตีความทำให้ backbone อ่อนถดถอย (gpt4o-mini G −0.51 t = −7.1; qwen G −0.44 t = −7.9) และ dissonance framing บน input เสียงชุดเดิมพลิกกลับเป็น +2.75 (t = 7.94) ⇒ "raw vocal metadata ไม่ therapeutic โดยตัวมันเอง; กรอบเหตุผลคือตัวแปลงสัญญาณเป็นคุณค่าทางคลินิก"; ส่วน noise จาก SER บน prosody สังเคราะห์เป็น limitation ที่ระบุไว้ | Tab 5.1/5.2; report F2; §6.3 |
| 7 | "ทดสอบเยอะขนาดนี้ p-hacking / multiple comparisons ไหม?" | เกณฑ์ตีความ **pre-registered ก่อนรัน pairwise** (p<0.05 + W>L = รองรับ; W≈L = ไม่ต่าง; p<0.05 + L>W = ค้าน); ตัวชี้วัดหลัก = composite 7 metric + pairwise win-rate, ราย metric เป็น exploratory; กฎ swap-consistency เป็น conservative (สองลำดับขัดกัน → tie); ทิศทาง effect สอดคล้องข้าม 3 ชั้นและข้าม judge | report §2 |
| 8 | "judge GPT-4o ตระกูลเดียวกับ backbone gpt4o-mini = self-enhancement?" | รับเป็น limitation ตรง ๆ; การบรรเทา: judge2 คนละครอบครัว (Claude) เห็นทิศทางตรงกันทุกเซลล์ที่ commit (และไม่เลือก Multimodal เลย 0/12 ในเซลล์ gpt4o-mini-vs-MM); effect หลักยังปรากฏบน backbone นอกตระกูล judge (DeepSeek pairwise p = 0.00004) | §5.4; limitations |
| 9 | "เสียงสังเคราะห์ TTS แล้ว generalize ไหม?" | แลก privacy/control กับ fidelity: Director กำหนดอารมณ์เสียงจากข้อความ ⇒ speech VA track text VA ใกล้ ⇒ flag ต่ำโดยโครงสร้าง; งานระบุ limitation ชัดและเสนอทางปิด: controlled mismatch manipulation + replay ด้วยเสียงมนุษย์จริง; ระหว่างนี้มีเดโมสคริปต์เสียงจริงแสดงเคส flag ทำงาน | §6.4 limitations; future work |
| 10 | "overhead +29% คุ้มไหม?" | +21.92 s/dialogue (75.38 → 97.33 s); ก้อนใหญ่สุดคือ Therapist LLM +1.64 s/ตอบ (response เชิงสำรวจยาวขึ้นโดยออกแบบ) + WavLM 0.25 s + librosa 0.09 s ต่อ utterance; dissonance computation เอง < 0.01 s; คุ้มสุดเมื่อจับคู่โมเดลเล็กต้นทุนต่ำซึ่งมี headroom มากสุด (Understanding 4.98 เทียบ Claude baseline 5.13 ที่เศษส่วนของต้นทุน) | §5.6 Tab 5.9/5.10 |
| 11 | "สรุป H1/H2 ผ่านไหม?" | **H1 partial**: บรรลุเกณฑ์ pre-registered "highest or tied-highest" บน GPT-4o mini (สูงสุดทั้ง 7 metric) และ DeepSeek (tied-highest + pairwise p = 0.00004); Qwen neutral เทียบ Baseline (รายงานตรง ๆ); Claude วัดไม่ได้ที่ ceiling **H2**: advantage เทียบ Baseline เป็น headroom-gated ไม่ใช่ architecture-independent แต่ robust core คือ ordering เทียบ condition ข้อมูลอื่น + ไม่แพ้ sig ใน pairwise layer ที่ใดเลย | §5.5 |

---

## ข้อห้าม/ข้อควรระวังตอนตอบ

- ห้ามพูดว่า "ชนะทุก backbone / ชนะทุกเงื่อนไข" — ใช้ "significant ที่ GPT-4o mini และ DeepSeek, neutral ที่ Qwen, ceiling ที่ Claude"
- ห้ามใช้คำ framing แบบ lie detection / deception / จับโกหก — ใช้ emotional dissonance, hidden affect, masked emotion, therapeutic exploration
- ถ้าถูกจี้ด้วยตัวเลขที่ดูเหมือนขัดกัน (เช่น t = −3.47 บน Claude) ให้ชี้ว่า **ชั้นไหนวัดอะไร**: absolute scale อิ่มตัว → pairwise คือ instrument แยกแยะ → cross-family judge คือตัวเช็คทิศทาง
- ทุกคำตอบปิดด้วย "นี่คือเหตุผลที่ออกแบบ evaluation 3 ชั้น + pre-registered criteria" เพื่อเปลี่ยนจุดอ่อนเป็นจุดแข็งทางวิธีวิทยา
