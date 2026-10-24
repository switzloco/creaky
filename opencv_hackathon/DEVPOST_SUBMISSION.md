# Devpost Submission Kit — OpenCV AI Competition 2026 (Powered by AWS)

## Project Title
**THE CMO (Chief MRI Officer) · Creaky-CV5: Autonomous 3-Plane Knee MRI Triage, OpenCV 5 Active-Zoom & Radiologist Second-Opinion Cockpit**

## Elevator Pitch (198 chars)
An autonomous OpenCV 5 (`dnn5` + AWS COOL) 3-plane Knee MRI agent (`0.957` RSNA LB) that sees across 144 slices, reasons over Cleanlab radiologist blind spots, and acts via 1.65× Lanczos4 focal zoom.

## Built With Tags
`opencv` · `aws` · `aws-graviton` · `python` · `onnx` · `pytorch` · `fastapi` · `html5` · `medical-imaging` · `agentic-ai`

## Live Web Endpoint & Repository Links
- **Instant Live Web Cockpit (Zero-Install Browser Endpoint):** `https://raw.githack.com/switzloco/creaky/main/opencv_hackathon/dist/index.html`
- **Interactive 3D Knee & UMAP Embedding Explorer ("Galaxy of Knees"):** `https://raw.githack.com/switzloco/creaky/main/galaxy_of_knees.html`
- **GitHub Repository:** `https://github.com/switzloco/creaky`
- **OpenCV 5 + AWS COOL Subdirectory:** `https://github.com/switzloco/creaky/tree/main/opencv_hackathon`
- **Target Prize Tracks:** Overall Grand Prize + **Best Use of COOL Award** + **Agentic Vision Award**

---

## 1. Problem Statement, Target Users & Real-World Impact

### The Clinical Bottleneck: 144 Slices × 3 Orthogonal Planes × 12 Pathologies
Reading a single musculoskeletal Knee MRI study requires a radiologist or orthopedic surgeon to mentally co-register **100 to 160 slices across three orthogonal planes** (`Sagittal`, `Coronal`, and `Axial`) to evaluate **12 interrelated pathologies** (`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `Patellofemoral OA`, `Joint Effusion`, `Synovitis`, `Baker's Cyst`, `Bone Contusion`, and `Cortical Fracture`).

During our development on the **RSNA Knee MRI Abnormality Detection Challenge (`2,668` multi-institutional studies)**, our domain and Confident Learning audit uncovered two critical failure modes that directly harm patient care:
1. **Human Single-Reader Blind Spots ("Satisfaction of Search", ~12–18% Omission Rate):** Comparing the `2,610` single-radiologist ("Silver") clinical reports against the 3-radiologist consensus ("Gold") panel revealed that busy single readers routinely omit secondary findings once a headline injury is spotted—such as missing a massive `6,981 px²` suprapatellar **joint effusion (`97.1%` probability)** after dictating an acute complete ACL rupture (`Case 01 / RSNA-267907`). Overall, clinical text reports omitted **77.6% of true Synovitis**, **44.8% of true Joint Effusions**, and **35.0% of subtle Meniscal Tears**.
2. **Passive Single-Pass 2D CNN Blind Spots:**
   - **Marrow Edema Confounder Storm (`Case 02 / RSNA-115480`):** Bright T2 subcortical bone marrow contusions bleed across standard convolutional feature maps, causing passive CNNs to hallucinate `>90%` tears across intact cruciate and collateral ligaments unless constrained by **subchondral cortical bone boundary gradients**.
   - **Focal Cartilage Delamination (`Case 03 / RSNA-100956`):** Standard `256×256` global pooling scored only `15.4%` on severe patellofemoral osteoarthritis because macro joint-space width appeared preserved, missing a narrow full-thickness median-ridge cartilage fissure.
   - **Grade II Intrameniscal Mucin vs. Operative Grade III Tear (`Case 04 / RSNA-101708`):** Without verifying the radiologic **Articular Surface Contact Rule**, raw pixel networks confuse benign internal mucoid degeneration with surface-extending tears requiring arthroscopic repair.

### Target Users & Dual-Mode Clinical Workflow
- **For Radiologists & Orthopedic Surgeons (`Doctor View`):** Acts as a `< 38 ms/slice` (`116.5 ms` per 144-slice 3D study on AWS Graviton COOL) second-opinion reader that flags omitted secondary pathologies, overlays OpenCV 5 cortical bone and contour biomarkers, and pairs a **1.65× `cv2.INTER_LANCZOS4` Active Focal Zoom** alongside an authentic **Healthy Knee MRI Reference** from the same anatomical plane.
- **For Patients (`Patient View`):** Translates dense radiological jargon into a plain-English **12-Structure Knee Pathology Review** with 3 tailored questions to ask their orthopedic specialist.

---

## 2. System Architecture & Agentic Workflow Diagram

```
====================================================================================================
     CREAKY-CV5 · AWS GRAVITON4 (COOL + KleidiCV) & OPENCV 5.0.0 AGENTIC VISION ARCHITECTURE
====================================================================================================

 [ Hospital PACS / Drag-and-Drop Knee MRI Volume (144 Slices: 48 Sag + 48 Cor + 48 Ax) ]
                                         │
                                         ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 0: HIPAA SAFE-HARBOR PIXEL PHI GUARDRAIL (`scrub_burned_in_phi_opencv5` · ~7.9 ms/study)  │
 │  • Anatomical tissue mask (`cv2.threshold` + `cv2.morphologyEx(MORPH_CLOSE)` + `findContours`)  │
 │  • Peripheral burned-in text glyph detection via White Top-Hat (`cv2.morphologyEx(MORPH_TOPHAT)`)│
 │  • Redacts peripheral MRN/DOB overlays in RAM & emits immutable `SHA-256` Safe-Harbor digest    │
 └───────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                         │ (0 Bytes PHI Egress · Ephemeral VPC RAM Buffer)
                                         ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ PASS 1 ("SEE"): OPENCV 5 VOLUMETRIC PRE-PERCEPTION & `dnn5` ONNX SLAB PROBING (AWS COOL ARM64)  │
 │  • FOV Air-Border Auto-Crop (`crop_fov_with_contours` -> `cv2.findContours` + `boundingRect`)   │
 │  • Dual-Res Scaling (`cv2.resize` 256×256 `INTER_AREA` + 384×384 `INTER_LINEAR` via KleidiCV)   │
 │  • 2.5D Slab Synthesis (`[z-1, z, z+1]`) -> `cv2.dnn.blobFromImage` + `cv2.dnn.readNetFromONNX` │
 │  • 3-Plane Anatomical MoE + Gated Attention Pool -> Per-Slice Attention α_{p,z} & 12 Logits     │
 └───────────────────────────────────────┬─────────────────────────────────────────────────────────┘
                                         │
                                         ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ PASS 2 ("REASON"): AGENTIC DIAGNOSTIC ORCHESTRATOR & CLEANLAB RADIOLOGIST AUDIT                 │
 │  • Selects Peak-Attention Slice z* = argmax_z(α_{p,z}) across Sagittal, Coronal & Axial stacks  │
 │  • Evaluates Cross-Resolution Tension (384px CoAtNet vs 256px ConvNeXt-S on Patellar Cartilage) │
 │  • Evaluates Confounder Risk (High Bone Contusion + Ligament Co-Activation -> Triggers Scharr)  │
 │  • Evaluates Meniscal Ambiguity (Grade II Mucin vs Grade III Tear -> Triggers Surface Contact)  │
 └───────────┬───────────────────────────────────────────┬─────────────────────────────────────────┘
             │ (Dynamic Tool Selection)                  │ (Human-in-the-Loop Steering)
             ▼                                           ▼
 ┌───────────────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
 │ PASS 3 ("ACT"): OPENCV 5 ACTIVE FOCAL PERCEPTION  │ │ INTERACTIVE DUAL-VIEWPORT PACS COCKPIT    │
 │  1. Rician Bilateral + CLAHE (`clipLimit=2.8`)    │ │  • Viewport A: Contour / CLAHE / Scharr   │
 │  2. Cortical Bone Edge (`cv2.Scharr` CV_16S x/y)  │ │  • Viewport B: 1.65× Zoom vs Healthy Knee │
 │  3. Hotspot Polygon (`cv2.findContours` + Moments)│ │  • Click-to-Steer Active Lens (~12 ms)    │
 │  4. Articular Surface Contact Rule (85th-pct Edge)│ │  • Doctor Second-Opinion + Patient Guide  │
 │  5. 1.65× `cv2.INTER_LANCZOS4` Active Focal Zoom  │ │  • 16-Slice 3D Attention Volume Scrubber  │
 └───────────────────────────────────────────────────┘ └───────────────────────────────────────────┘
```

---

## 3. OpenCV 5 Implementation Deep-Dive

Every stage of **Creaky-CV5** ([`opencv_hackathon/cv5_engine.py`](https://github.com/switzloco/creaky/blob/main/opencv_hackathon/cv5_engine.py)) is powered by native **OpenCV 5** (`cv2`) computer-vision primitives rather than treating vision as a black-box wrapper:

1. **Automated Burned-In Pixel PHI Scrubbing (`scrub_burned_in_phi_opencv5`):**
   - Legacy MRI scanners and secondary capture PACS exports frequently burn patient identifiers (MRN, Name, DOB) directly into the outer pixel margins.
   - We isolate the primary anatomical knee contour using `cv2.threshold(gray, 14, 255, cv2.THRESH_BINARY)`, elliptical morphological closing (`cv2.morphologyEx(..., cv2.MORPH_CLOSE)`), `cv2.findContours`, and `cv2.dilate`.
   - Outside the core joint anatomy, a rectangular **Morphological White Top-Hat (`cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, (11,3))`)** isolates high-frequency alphanumeric strokes, redacting burned-in text glyphs in `~0.05 ms/slice` and generating a cryptographic `SHA-256` Safe-Harbor digest.
2. **FOV Air-Border Auto-Cropping (`crop_fov_with_contours`):**
   - Uses `cv2.findContours` + `cv2.boundingRect` on morphologically closed tissue masks to strip non-diagnostic background air borders before resizing, increasing effective pixel resolution across the tibiofemoral and patellofemoral joint spaces by `~28%`.
3. **Multi-Resolution 2.5D Slab Synthesis & `cv2.dnn` (`dnn5`) ONNX Probing:**
   - Generates parallel `256×256` (`cv2.INTER_AREA`) and `384×384` (`cv2.INTER_LINEAR`) slices stacked into `[z-1, z, z+1]` 2.5D volumetric slabs.
   - Feeds slabs through `cv2.dnn.blobFromImage` into `cv2.dnn.readNetFromONNX("creaky_slice_probe.onnx")` to score slice-level pathology salience natively inside OpenCV's C++ graph engine.
4. **Rician-Denoised Adaptive Joint-Space CLAHE (`enhance_slice_multimodal`):**
   - Applies edge-preserving bilateral filtering (`cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)`) to suppress MRI Rician coil noise while preserving sharp meniscal margins, followed by `cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))`.
5. **Cortical Bone & Articular Surface Scharr Gradient Mapping:**
   - Computes directional first derivatives via `cv2.Scharr(clahe_gray, cv2.CV_16S, 1, 0)` and `cv2.Scharr(clahe_gray, cv2.CV_16S, 0, 1)`, fused via `cv2.convertScaleAbs` and `cv2.addWeighted` to delineate the high-gradient subchondral cortical bone plate.
6. **Hotspot Contour Segmentation, Spatial Moments & 1.65× `cv2.INTER_LANCZOS4` Active Focal Zoom:**
   - Thresholds top-percentile spatial activation, applies morphological opening/closing, extracts polygon contours via `cv2.findContours`, computes the exact lesion centroid $(c_x, c_y) = (M_{10}/M_{00}, M_{01}/M_{00})$ via `cv2.moments`, and crops/upsamples both an annotated ROI and an unmarked high-contrast tissue patch at `1.65×` magnification using 8×8 Lanczos-4 interpolation (`cv2.INTER_LANCZOS4`).

---

## 4. Additional Evidence for the Agentic Vision Award ($1,000 Track)

To qualify for the **Agentic Vision Award**, OpenCV 5 visual measurements must actively change a subsequent plan, tool call, diagnostic classification, or human-in-the-loop escalation. Below are **three concrete execution traces** built into Creaky-CV5:

### Trace A: Suppressing False-Positive Ligament Surgery via `cv2.Scharr` Cortical Containment (`Case 02 / RSNA-115480`)
- **Pass 1 (`SEE`):** Volumetric slab probing detects intense T2 hyperintensity in the lateral knee compartment (`Bone Contusion: 94.8%`, `Joint Effusion: 96.2%`). In naive 2D CNNs, this bright marrow edema bleeds across receptive fields and triggers false-positive ACL/MCL tear alerts.
- **Pass 2 (`REASON`):** Because `Contusion > 85%` co-occurs with adjacent ligament proximity, the agent schedules the **OpenCV 5 Cortical Containment Verification tool** (`enhance_slice_multimodal` + `agentic_focal_zoom_and_contours`).
- **Pass 3 (`ACT` & Decision Change):** `cv2.Scharr` measures a high cortical boundary gradient (`167.1`) enclosing the `6,981 px²` hyperintense contour, verifying that the edema is strictly confined **inside the subchondral bone plate** rather than traversing the intercondylar ligament fibers.
- **Downstream Outcome:** The agent suppresses false-alarm ligament probabilities (`ACL: 18.4%`, `MCL: 15.2%`), switches the clinical badge to **`FALSE-ALARM PREVENTED`**, and routes the patient to **non-operative protected weight-bearing** instead of an unnecessary orthopedic surgical referral.

### Trace B: Differentiating Grade III Operative Meniscal Tear vs. Grade II Internal Mucin via the Articular Surface Contact Rule (`Case 04 / RSNA-101708`)
- **Pass 1 (`SEE`):** Coronal fluid-sensitive MRI shows a linear high-signal focus inside the lateral meniscus (`91.6%` raw anomaly score).
- **Pass 2 (`REASON`):** In musculoskeletal radiology, an intrameniscal signal is only a true **Grade III tear** if it extends to and contacts the superior or inferior **articular surface cortex**; internal-only signal is benign Grade II mucoid degeneration. The agent invokes OpenCV 5 perimeter-edge intersection analysis.
- **Pass 3 (`ACT` & Decision Change):** The agent constructs a 3-pixel perimeter ring around the segmented lesion polygon (`cv2.drawContours(cnt_ring, [primary_cnt], -1, 255, thickness=3)`) and intersects it against the `cv2.Scharr` cortical gradient map (`mm["scharr_mag"][cnt_ring > 0]`). Measuring an **85th-percentile articular surface contact score of `99.4%`** (exceeding the `75.0%` operative threshold) and centering the `1.65× cv2.INTER_LANCZOS4` lens at `(249, 255)`, the agent confirms surface extension and escalates the case for **arthroscopic meniscal repair evaluation**.

### Trace C: Interactive Physician Click-to-Steer Active Focal Lens (`/api/agent_zoom`)
- **Human-in-the-Loop Control:** At any point during review, the physician can click any anatomical coordinate $(x, y)$ on Viewport A (`left-canvas`).
- **Real-Time Re-Execution (`~12–34 ms`):** The click coordinates are normalized and dispatched to `/api/agent_zoom`, which dynamically re-centers the Gaussian spatial prior, re-runs `cv2.findContours` + `cv2.moments`, recomputes tissue biomarkers (`ROI Area`, `Articular Surface Contact %`, `Cortical Edge Clarity`), and updates Viewport B with a freshly interpolated `1.75× cv2.INTER_LANCZOS4` focal zoom.

---

## 5. Additional Evidence for the Best Use of COOL Award ($1,000 Track)

### AWS Graviton + OpenCV COOL Deployment Configuration
- **Target Cloud Instance:** AWS EC2 `c7g.xlarge` (AWS Graviton3, Neoverse-V1) and `c8g.xlarge` (AWS Graviton4, Neoverse-V2) running **Ubuntu 24.04 LTS (`aarch64`)** with the **Cloud-Optimized OpenCV Library (COOL / OpenCV 5.0.0 + Arm KleidiCV NEON/SVE2 acceleration)** from AWS Marketplace.
- **Hybrid x86 / Arm Coexistence Architecture:**
  - `opencv_hackathon/cv5_engine.py` automatically detects the hardware architecture (`platform.machine()`) and SIMD dispatch flags (`cv2.getBuildInformation()`).
  - On **AWS Graviton ARM64 (`aarch64`)**, KleidiCV accelerates the heaviest per-slice inner loops—specifically `cv2.morphologyEx` (`MORPH_TOPHAT` / `MORPH_CLOSE`), `cv2.resize` (`INTER_AREA` / `INTER_LINEAR` / `INTER_LANCZOS4`), `cv2.bilateralFilter`, `cv2.createCLAHE`, and `cv2.Scharr` 16-bit signed convolution—while `cv2.dnn.readNetFromONNX` executes the 2.5D slab probe on Arm NEON/FP16 cores without requiring an expensive GPU instance.
- **1-Command Reproducible Deployment (`opencv_hackathon/deploy_aws_cool.sh`):**
  ```bash
  bash opencv_hackathon/deploy_aws_cool.sh
  ```

### Reproducible Benchmark: x86_64 Baseline vs. AWS Graviton COOL (144-Slice 3-Plane Study)
You can reproduce this exact 6-stage benchmark on any machine via `uv run python -c "from opencv_hackathon.cv5_engine import run_opencv_cool_benchmark; import json; print(json.dumps(run_opencv_cool_benchmark(144), indent=2))"` or by clicking **`⚡ Re-Run Live 144-Slice Benchmark`** inside the Cockpit footer:

| Pipeline Stage (144 Slices: 48 Sag + 48 Cor + 48 Ax) | OpenCV 5 API Executed | x86_64 Baseline (`c6i.xlarge`) | AWS Graviton COOL (`c7g.xlarge` Arm64) | Speedup / Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **0. HIPAA Safe-Harbor Pixel PHI Scrub** | `cv2.morphologyEx(MORPH_TOPHAT)` + `threshold` | `14.2 ms` | `7.9 ms` | **1.80× faster** |
| **1. FOV Air-Border Auto-Crop** | `cv2.threshold` + `MORPH_CLOSE` + `findContours` | `22.3 ms` | `12.5 ms` | **1.78× faster** |
| **2. Dual-Resolution Scaling (256 + 384)** | `cv2.resize` (`INTER_AREA` + `INTER_LINEAR`) | `19.1 ms` | `10.7 ms` | **1.79× faster** |
| **3. Adaptive Joint-Space CLAHE** | `cv2.bilateralFilter` + `cv2.createCLAHE(2.8)` | `16.8 ms` | `9.4 ms` | **1.79× faster** |
| **4. Cortical Bone Scharr Edge + LUT** | `cv2.Scharr(CV_16S)` + `addWeighted` + `TURBO` | `80.3 ms` | `45.0 ms` | **1.78× faster** |
| **5. 2.5D Slab Neural Probe (`dnn5`)** | `cv2.dnn.blobFromImage` + `readNetFromONNX` | `55.4 ms` | `31.0 ms` | **1.79× faster** |
| **Total 144-Slice 3-Plane Knee MRI Study** | **End-to-End 6-Stage Pipeline** | **`208.1 ms` (`692 sl/s`)** | **`116.5 ms` (`1,236 sl/s`)** | **1.79× Throughput** |
| **Compute Cost per 10,000 MRI Studies** | **On-Demand EC2 Instance Cost (`$0.17/hr` vs `$0.145/hr`)** | **`$0.098 / 10k studies`** | **`$0.047 / 10k studies`** | **52.0% Lower Cloud Cost** |

---

## 6. Quantitative Evaluation, Failure Cases & Limitations

### Principal Evaluation Results (`2,668` Multi-Institutional RSNA Knee MRI Studies)
We evaluated our multi-resolution 3-plane architecture across three rigorous tiers:
1. **RSNA Public Leaderboard (`1,500` held-out test studies):** **`0.957` Macro ROC-AUC** (Cleanlab-pruned Multi-Resolution 3-Plane Ensemble: `384×384` CoAtNet + `256×256` ConvNeXt-Small Anatomical MoE with N4 Bias-Field Correction and multilingual report-supervision auxiliary head).
2. **5-Fold Out-of-Fold Silver Validation (`2,610` single-reader studies):** **`0.8676` Macro ROC-AUC**.
3. **Independent 3-Radiologist Gold Consensus Panel (`58` multi-expert studies):** **`0.8568` Macro ROC-AUC**.

| Pathology Target | Silver OOF ROC-AUC (`n=2,610`) | Clinical Text Report Coverage (Gold Audit) | Primary OpenCV 5 Biomarker Leveraged |
| :--- | :---: | :---: | :--- |
| **Baker's Cyst** | `0.9090` | `81.2%` | Popliteal fossa fluid contour area (`cv2.findContours`) |
| **Medial Compartment OA** | `0.8928` | `69.6%` | `cv2.Scharr` subchondral cortical edge + `1.65×` zoom |
| **Medial Meniscus Tear** | `0.8717` | `65.0%` | Articular surface contact % (`cv2.Scharr` ∩ contour ring) |
| **Cortical Fracture** | `0.8690` | `88.5%` | Cortical step-off discontinuity (`cv2.Scharr` gradient) |
| **ACL Rupture** | `0.8629` | `91.4%` | Intercondylar fiber continuity (`cv2.createCLAHE` + zoom) |
| **Lateral Compartment OA** | `0.8617` | `68.0%` | Lateral tibiofemoral joint-space `cv2.Scharr` margin |
| **Joint Effusion** | `0.8574` | `55.2%` (**44.8% omitted**) | Suprapatellar fluid hyperintensity Z-score (`Z = +2.4`) |
| **Patellofemoral OA (PF OA)** | `0.8523` | `64.1%` | `384px` + `1.65× INTER_LANCZOS4` median-ridge zoom |
| **Synovitis** | `0.8367` | `22.4%` (**77.6% omitted**) | Hoffa's fat-pad hyperintense contour segmentation |
| **Bone Contusion** | `0.8322` | `71.0%` | Subchondral marrow edema containment inside `cv2.Scharr` |
| **MCL Injury** | `0.8294` | `78.4%` | Coronal medial collateral fiber contour tracking |
| **Lateral Meniscus Tear** | `0.7901` | `65.0%` | Coronal/Sagittal peripheral slice sampling (see Limitations) |

### Documented Failure Cases & Technical Limitations
In accordance with clinical transparency best practices, we document three specific failure modes uncovered during our error audit (`doctor_mri_error_audit.ipynb`):
1. **Peripheral Sagittal Truncation of the Lateral Meniscus (`0.7901` AUC — Lowest Target):**
   - *Failure Mechanism:* To accelerate 3D volumetric caching, our baseline slice sampler extracted the central `12%–88%` of slices in each stack (`K=16` evenly spaced slices). While the ACL sits dead-center in the intercondylar notch (`~50%` sagittal depth), the **body and free edge of the lateral meniscus** sit at the extreme lateral periphery (`8%–18%` or `82%–92%` sagittal depth). In wide-FOV scans, central-band truncation occasionally clips the lateral meniscal body.
   - *Mitigation:* Cross-plane attention fusion (`CrossPlaneTransformer`) partially recovers lateral meniscal tears from the **Coronal** plane, and our OpenCV 5 FOV contour auto-crop (`crop_fov_with_contours`) tightens the anatomical bounding box prior to slice selection.
2. **Post-Operative Metallic Hardware Susceptibility Artifacts:**
   - *Failure Mechanism:* Patients with prior ACL interference screws or total knee arthroplasty hardware exhibit magnetic susceptibility blooming (signal void surrounded by a bright pile-up ring). This artificial high-contrast ring produces steep gradients in `cv2.Scharr`, which can inflate the `cortical_edge_gradient` and `articular_surface_contact_pct` biomarkers.
   - *Mitigation:* Our Cockpit always pairs the AI contour view with an **unmarked `1.65× Clean Tissue Zoom`** (`clean_zoom_b64`) so the reviewing physician can visually recognize metallic blooming artifacts immediately.
3. **MRI "Magic Angle" Artifact ($55^\circ$ to $B_0$) in Short-TE Sequences:**
   - *Failure Mechanism:* Collagen fibers oriented at $\approx 55^\circ$ to the main magnetic field $B_0$ (particularly the posterior horn of the lateral meniscus and patellar tendon on short-TE Proton Density sequences) exhibit artificial T2/PD hyperintensity that mimics a mucoid tear.
   - *Mitigation:* The system requires multi-plane corroboration across both Sagittal and Coronal stacks before raising high-confidence meniscal alerts.

---

## 7. Responsible-Use, Clinical Safety & HIPAA Governance

1. **Human-in-the-Loop Clinical Decision Support (Non-Autonomous CDS):**
   - **Creaky-CV5** is strictly designed as an **assistive second-opinion triage and visualization tool**, not an autonomous diagnostic device. Every finding is presented alongside the original unmarked MRI slice, the enhanced `1.65×` tissue zoom, and an authentic healthy knee reference so a licensed radiologist or orthopedic specialist retains 100% final diagnostic authority.
2. **Automated Burned-In Pixel PHI Scrubbing (`45 CFR § 164.514` Safe-Harbor):**
   - Beyond stripping 18 standard DICOM header tags, `scrub_burned_in_phi_opencv5` automatically detects and black-redacts burned-in alphanumeric text overlays in the outer image margins using `cv2.morphologyEx(cv2.MORPH_TOPHAT)` before any slice enters the neural/vision pipeline, emitting a verifiable `SHA-256` Safe-Harbor digest (`SH256-...`).
3. **Zero-Egress AWS COOL VPC Operation:**
   - All image decoding (`cv2.imdecode`), PHI scrubbing, CLAHE/Scharr enhancement, and `cv2.dnn` ONNX inference execute entirely in **ephemeral RAM buffers** inside the hospital's isolated AWS Graviton VPC instance. Zero patient pixels are ever written to disk or transmitted to external third-party LLM APIs.

---

## 8. Pinned Dependencies, Build, Deployment & Test Instructions

### Quick Start (Local x86_64 or AWS Graviton ARM64)
```bash
# 1. Install pinned dependencies (or run deploy_aws_cool.sh on AWS EC2 Graviton)
uv pip install -r opencv_hackathon/requirements.txt

# 2. Run the automated unit & benchmark verification suite
uv run python -m unittest tests/test_opencv_hackathon.py -v

# 3. Launch the interactive FastAPI + OpenCV 5 Cockpit on http://localhost:8055
uv run python -m opencv_hackathon.app

# 4. (Optional) Export the 100% standalone static HTML bundle for instant serverless hosting
uv run python -m opencv_hackathon.app --export-static
```
