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
- **Reporting Bias / Selective Omission (hypothesis, not yet measured):** Radiologists tend to report what the clinical indication asked about. With a referral like "acute twist / suspected ACL tear", mild effusion, minor synovitis or subtle PF cartilage thinning may go unmentioned even when visible on MRI.
  - *Correction:* the earlier "30–60% of gold positives are not mentioned in the report" figure came from `scripts/audit_gold_disagreements.py`, which measures the **fallback regex labeler**, not the reports and not Jev. A regex "miss" can just mean the report phrased it in a way (or a language) the regex doesn't know. So that number is an upper bound that mixes regex misses with real omissions.
  - *How to measure it properly:* `scripts/label_mining/audit_jev_vs_gold.py` (Jev vs gold, per target, with counts). Results:

```
Gold studies: 58 | Jev rows: 4407 | threshold: 0.5
          Target  Gold pos  Gold neg Jev missed (pos->neg)  Miss rate Jev false alarms (neg->pos)  False alarm rate  Jev-negative precision (NPV)  Jev unsure (|p-0.5|<band)  Few positives
             ACL        24        34                  0/24       0.00                        8/34              0.24                          1.00                          3          False
             MCL         9        49                   0/9       0.00                        7/49              0.14                          1.00                          2          False
 Medial Meniscus        26        32                  1/26       0.04                        9/32              0.28                          0.96                          3          False
Lateral Meniscus        23        35                  3/23       0.13                       10/35              0.29                          0.89                          1          False
       Medial OA        15        43                  1/15       0.07                        4/43              0.09                          0.97                          3          False
      Lateral OA        11        47                  3/11       0.27                        6/47              0.13                          0.93                          2          False
           PF OA        21        37                  8/21       0.38                        2/37              0.05                          0.81                          9          False
        Effusion        35        23                  0/35       0.00                       16/23              0.70                          1.00                          0          False
       Synovitis        27        31                 11/27       0.41                        7/31              0.23                          0.69                         15          False
         Baker's        12        46                  1/12       0.08                        5/46              0.11                          0.98                          0          False
       Contusion        19        39                  2/19       0.11                       19/39              0.49                          0.91                          4          False
        Fracture        18        40                  4/18       0.22                        6/40              0.15                          0.89                          2          False
```
- **Contusion vs. Soft Tissue Edema:** Text regex on "edema" or "contusion" produces false positives when the report discusses subcutaneous or muscular swelling rather than osseous (bone marrow) contusion.
- **Indirect Meniscal Phrasing:** Radiologists frequently omit the word "tear" and use functional descriptions: *"signal reaches inferior articular surface"*, *"blunted posterior horn"*, *"degenerative meniscopathy"*, or *"maceration"*.
- **Synovitis Ambiguity:** Radiologists rarely diagnose "synovitis" by name on unenhanced knee MRI without IV gadolinium contrast; instead they describe "joint fluid accumulation", "plica irritation", or "Hoffa fat pad edema".

## Evaluating like a med-device team
- **Label Uncertainty Masking:** If a target's report-derived negatives often turn out to be unmentioned findings rather than confirmed absences, those negatives should carry less weight in the loss than confirmed ones. Whether that's true, and for which targets, depends on the Jev-vs-gold audit above. With 58 gold studies, rates for targets with fewer than ~5 gold positives are anecdotes.
- **Multi-Site Generalization:** Reports in English, Spanish, Bulgarian, and Greek reflect different international dictation conventions; models must rely on anatomical image features rather than report artifacts.

## Laterality and Anatomical Mirror-Symmetry (Task T3 Audit)
- **Natural Left/Right Distribution:** In `train.csv` (4,407 studies), clinical report laterality indicates 54.6% Right knees (1,002 explicit) and 45.4% Left knees (834 explicit) among single-knee determined studies.
- **Anatomical Mirror Principle:** Medial vs Lateral anatomy is defined intrinsic to the knee joint (e.g. the fibula is lateral, tibia plateau medial; MCL medial, LCL lateral), NOT by screen coordinates or 2D image side.
- **Validity of Horizontal Flip:** Because the dataset natively comprises a near-even 55/45 mix of right and left knees, reflecting a right knee horizontally creates a anatomically plausible left knee (and vice versa) with identical pathologies mapped to identical compartments. Horizontal flip (p=0.5) is anatomically non-destructive, provided it is applied consistently across all slices and planes without flipping labels.

## Lateral Meniscus Diagnostic Deep-Dive (Task T8 Audit)
- **Why is Lateral Meniscus the Lowest-Scoring Target (~0.77 - 0.79 AUC)?**
  1. **Clinical Nuance & Grade II vs. III Degeneration:** In Turkish/German/Spanish dictations (e.g. *"grade II dejenerasyon"*, *"amputación marginal"*), radiologists distinguish intrasubstance degeneration from a true frank tear reaching the articular margin. Jev frequently assigns 0.85–0.97 probability to non-tear degeneration, causing 10 false alarms across the 58 Gold studies.
  2. **Co-occurring Medial Tears:** 8 of the 10 false alarm studies have a true Medial Meniscus tear. In dictations describing both compartments, Jev's attention mechanism leaks the Medial tear finding into the Lateral compartment probability.
  3. **Anatomical Eccentricity in Sagittal Slices:** Unlike the medial meniscus which has a wide bow-tie cross-section centrally, the lateral meniscus is smaller, circular, and lies further out on the lateral sagittal periphery. When coronal/sagittal planes are cropped to the central 12–88% volume band (`cache_v1`), the most peripheral slices showing the lateral meniscal root can be clipped.

## Open questions
-
