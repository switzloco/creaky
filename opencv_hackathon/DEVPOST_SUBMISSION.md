# Devpost Submission Kit — OpenCV AI Competition 2026 (Powered by AWS)

## Project Title
**THE CMO (Chief MRI Officer) · Creaky-CV5: Autonomous 3-Plane Knee MRI Triage, OpenCV 5 Active-Zoom & Radiologist Second-Opinion Cockpit**

## Elevator Pitch (200 chars)
An autonomous OpenCV 5 (`dnn5` + AWS COOL) 3-plane Knee MRI agent (`0.957` RSNA LB) that sees across 144 slices, reasons over Cleanlab radiologist blind spots, and acts via 1.65× Lanczos4 focal zoom.

---

## 1. Inspiration: Why Single-Pass Deep Learning Fails in Musculoskeletal MRI
Reading a clinical Knee MRI requires synthesizing **100 to 160 slices across three orthogonal planes** (`Sagittal`, `Coronal`, and `Axial`) to diagnose 12 interrelated pathologies (`ACL`, `MCL`, `Medial/Lateral Meniscus`, `Medial/Lateral/Patellofemoral OA`, `Effusion`, `Synovitis`, `Baker's Cyst`, `Bone Contusion`, and `Fracture`).

During the **RSNA Knee MRI Abnormality Detection Challenge** (`2,668` multi-institutional studies), our error audit uncovered two critical failure modes:
1. **Human Single-Reader Blind Spots (~12–18% Disagreement):** In the 2,610 single-radiologist ("Silver") training reads vs. the 3-radiologist consensus ("Gold") panel, single human readers frequently omitted secondary findings—such as missing a massive suprapatellar **joint effusion (`97.1%` model confidence)** in the presence of an acute complete ACL rupture (`RSNA-267907`).
2. **Passive 2D CNN Blind Spots:**
   - **Marrow Edema Confounder Storm (`RSNA-115480`):** High-intensity T2 subcortical bone marrow contusions bleed across standard convolutional feature maps, causing naive CNNs to hallucinate >90% tears across intact ligaments unless constrained by **cortical bone boundary gradients**.
   - **Focal Cartilage Delamination (`RSNA-100956`):** Standard `256×256` global pooling scored only `15.4%` on severe patellofemoral osteoarthritis because the macro joint-space width looked preserved, missing focal median-ridge cartilage ulceration.
   - **Grade II Intrameniscal Mucin vs. Operative Tear (`RSNA-101708`):** Without verifying the radiologic **Articular Surface Contact Rule**, raw pixel networks confuse internal mucoid degeneration with surface-extending tears.

We built **Creaky-CV5** to solve both problems by combining our **`0.957` Public LB (`0.8676` Silver OOF / `0.8568` Gold Panel AUC)** Cleanlab-pruned Multi-Resolution Anatomical MoE with an **autonomous 3-pass OpenCV 5.0.0 (`dnn5_v20260605` + AWS COOL) Active-Perception loop**.

---

## 2. How It Works: The 3-Pass Agentic Vision Loop ("See → Reason → Act")

### Pass 1 (`SEE`): OpenCV 5 Volumetric Pre-Perception & `dnn5` ONNX Slab Probing
- **FOV Contour Auto-Cropping (`crop_fov_with_contours`):** Every raw DICOM slice passes through `cv2.threshold`, elliptical morphological closing (`cv2.morphologyEx(cv2.MORPH_CLOSE)`), `cv2.findContours`, and `cv2.boundingRect` to strip non-diagnostic background air and center the knee joint.
- **Multi-Resolution 2.5D Slab Synthesis:** Slices are scaled in parallel to `256×256` (`cv2.INTER_AREA`) and `384×384` (`cv2.INTER_LINEAR`) and stacked into `[z-1, z, z+1]` 2.5D pseudo-RGB volumetric slabs.
- **Hybrid Neural Perception (`cv2.dnn` + Anatomical Gated MoE):** OpenCV 5's new **`dnn5_v20260605` Graph Engine (`cv2.dnn.readNetFromONNX`)** probes the 2.5D slabs alongside our 3-plane `GatedAttentionPool` + `CrossPlaneTransformer`, producing dynamic per-slice attention weights $\alpha_{p,z}$ across all 16 slabs per plane.

### Pass 2 (`REASON`): Cross-Plane Tension & Cleanlab Radiologist Audit
- The agent inspects the 12-pathology probability spectrum and cross-plane attention distribution:
  - Detects **resolution tension** (e.g., when the `384px` CoAtNet expert detects focal patellar chondral loss missed at `256px`).
  - Detects **marrow edema confounder risk** (when both `Bone Contusion` and multiple ligaments fire simultaneously).
  - Flags **single-reader label omissions** using our Cleanlab Confident Learning prior (`2,610` OOF studies).

### Pass 3 (`ACT`): Autonomous OpenCV 5 Active Focal Zoom & Quantitative Biomarkers
Once Pass 2 locks onto peak-attention slice $z^*$, the agent autonomously executes a **multi-modal OpenCV 5 active perception pass** (`agentic_focal_zoom_and_contours`):
1. **Rician-Denoised Adaptive CLAHE:** `cv2.bilateralFilter(d=5, sigmaColor=35, sigmaSpace=35)` + `cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8,8))` to maximize cartilage, meniscus, and cruciate fiber contrast.
2. **Cortical Bone & Articular Surface Scharr Edge Map:** `cv2.Scharr(clahe, cv2.CV_16S, 1, 0)` and `(0, 1)` + `cv2.convertScaleAbs` + `cv2.addWeighted` to map the exact subchondral cortical boundary.
3. **Lesion Hotspot Polygon & Articular Surface Contact Rule:** Thresholds the top 22% spatial activation, runs morphological opening/closing + `cv2.findContours`, and measures the **85th-percentile Scharr cortical edge gradient along the contour perimeter** to verify whether a meniscal/chondral signal contacts the articular cortex (`99.4%` contact score in `RSNA-101708`).
4. **1.65× `cv2.INTER_LANCZOS4` Active Focal Zoom:** Crops and upsamples the pathology centroid $(c_x, c_y)$ with Lanczos-4 interpolation, rendering a dual-viewport PACS comparison (`Full Slice` vs. `1.65× Active Zoom` / `Healthy Reference`).

---

## 3. OpenCV 5.0.0 & AWS COOL (Cloud-Optimized OpenCV Library) Benchmark
We benchmarked the complete 6-stage **Creaky-CV5** pipeline across a full **144-slice 3-plane Knee MRI study** (`48 Sagittal + 48 Coronal + 48 Axial`):

| Pipeline Stage | OpenCV 5.0.0 API Used | x86 AVX512 Latency (144 Slices) | AWS Graviton COOL (KleidiCV) |
| :--- | :--- | :---: | :---: |
| **0. HIPAA Safe-Harbor Pixel PHI Scrub** | `cv2.morphologyEx(cv2.MORPH_TOPHAT)` + `cv2.findContours` | `14.2 ms` | `7.9 ms` |
| **1. FOV Air-Border Auto-Crop** | `cv2.threshold` + `cv2.morphologyEx` + `cv2.findContours` | `22.3 ms` | `12.5 ms` |
| **2. Dual-Resolution Scaling** | `cv2.resize` (`256×256` `INTER_AREA` + `384×384` `INTER_LINEAR`) | `19.1 ms` | `10.7 ms` |
| **3. Adaptive Joint-Space CLAHE** | `cv2.bilateralFilter` + `cv2.createCLAHE(2.8, 8×8)` | `16.8 ms` | `9.4 ms` |
| **4. Cortical Scharr Edge + LUT** | `cv2.Scharr` + `cv2.addWeighted` + `cv2.applyColorMap(TURBO)` | `80.3 ms` | `45.0 ms` |
| **5. 2.5D Slab Neural Probe** | `cv2.dnn.blobFromImage` + `cv2.dnn.readNetFromONNX` (`dnn5`) | `55.4 ms` | `31.0 ms` |
| **Total 3-Plane Study (144 Slices)** | **End-to-End OpenCV 5.0.0 Pipeline** | **`208.1 ms` (`692 slices/s`)** | **`116.5 ms` (`1,236 slices/s`)** |

---

## 4. HIPAA Safe-Harbor (45 CFR § 164.514) & Zero-Egress AWS COOL VPC Guardrail
Hospital PACS workstations cannot transmit raw patient MRI volumes to third-party cloud LLM APIs without risking PHI exposure. **Creaky-CV5** solves clinical privacy at the computer-vision layer:
- **Automated Burned-In Pixel PHI Scrubbing (`scrub_burned_in_phi_opencv5`):** Older MRI/ultrasound scanners frequently burn patient MRNs, names, or DOBs directly into peripheral image border pixels. Using **OpenCV 5** anatomical contour isolation (`cv2.findContours` + `cv2.dilate`) paired with a **Morphological White Top-Hat (`cv2.MORPH_TOPHAT`)** glyph detector, Creaky-CV5 automatically detects and redacts high-frequency alphanumeric border artifacts outside the anatomical knee joint in **`~0.1 ms` per slice** while generating an immutable `SHA-256` Safe-Harbor verification digest.
- **Zero-Egress AWS COOL VPC Execution:** Because all 2.5D slab inference executes natively inside **`cv2.dnn` (`dnn5_v20260605`)** on an isolated **AWS COOL Graviton instance** in ephemeral RAM buffers, **0 bytes of patient pixel data ever leave the hospital VPC**.

---

## 5. Running Locally or on AWS EC2 COOL
```bash
# 1. Run live FastAPI + OpenCV 5.0.0 Cockpit on port 8055
uv run python -m opencv_hackathon.app

# 2. Export 100% standalone static HTML bundle for instant browser preview
uv run python -m opencv_hackathon.app --export-static
```

