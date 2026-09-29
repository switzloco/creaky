# Domain Notes — Knee MRI, Sports Medicine, Med Devices

A running notebook of what's been learned about the domain (not the code). A few lines at a time is enough. Add a source or the study/report that taught it where you can.

## Anatomy and injuries
- **ACL (Anterior Cruciate Ligament):** Runs diagonally through intercondylar notch from posterior lateral femur to anterior medial tibia. 45° angle best seen on Sagittal view. ACL tear often causes anterior tibial translation and kissing contusions on lateral femoral condyle / posterior lateral tibial plateau.
- **MCL (Medial Collateral Ligament):** Broad, flat band on inner medial knee stabilizing against valgus stress. Best seen on Coronal view spanning from medial femoral epicondyle to proximal tibia.
- **Menisci (Medial & Lateral):** Fibrocartilaginous shock absorbers. Classic "bow-tie" appearance on Sagittal peripheral slices, triangular wedges on central slices. Normal meniscus is homogeneously black (low signal) on all pulse sequences. Pathologic tears show linear high T2/PD signal communicating with the superior or inferior articular surface.

## MRI planes and sequences
- **Sagittal:** Gold standard for Cruciate Ligaments (ACL/PCL), Meniscal anterior and posterior horns, and extensor mechanism (patellar tendon).
- **Coronal:** Crucial for collateral ligaments (MCL, LCL), medial/lateral joint space loss (Osteoarthritis), and meniscal body segments.
- **Axial:** Optimal for Patellofemoral joint (PF OA), trochlear groove dysplasia, joint effusion, and synovial thickening / synovitis.

## How reports describe findings (Phase 3 Audit Findings)
- **Reporting Bias / Selective Omission:** Radiologists report what clinical indication requested. If referral is "acute twist / suspected ACL tear", doctors often omit mild effusion, minor synovitis, or subtle PF cartilage thinning even when visibly present on MRI. This causes 30–60% of positive Gold cases to have no explicit mention in the text report!
- **Contusion vs. Soft Tissue Edema:** Text regex on "edema" or "contusion" produces false positives when the report discusses subcutaneous or muscular swelling rather than osseous (bone marrow) contusion.
- **Indirect Meniscal Phrasing:** Radiologists frequently omit the word "tear" and use functional descriptions: *"signal reaches inferior articular surface"*, *"blunted posterior horn"*, *"degenerative meniscopathy"*, or *"maceration"*.
- **Synovitis Ambiguity:** Radiologists rarely diagnose "synovitis" by name on unenhanced knee MRI without IV gadolinium contrast; instead they describe "joint fluid accumulation", "plica irritation", or "Hoffa fat pad edema".

## Evaluating like a med-device team
- **Label Uncertainty Masking:** Because 30–50% of negative reports represent "unmentioned / omitted" findings rather than confirmed absence of disease, treating unmentioned findings as soft labels (`weight=0.3` or `0.0`) is medically superior to forcing binary 0.0 cross-entropy loss.
- **Multi-Site Generalization:** Reports in English, Spanish, Bulgarian, and Greek reflect different international dictation conventions; models must rely on anatomical image features rather than report artifacts.

## Open questions
-
