# Moonshot & Novel Ideas Backlog

This backlog tracks novel, out-of-the-box ideas to explore during the iteration phase (Weeks 8–10).

---

## 1. Arthroscopy & Video Pretraining / Cross-Modal Alignment
- **Idea:** Compare or pretrain representations using arthroscopy surgical videos / anatomical footage (e.g., YouTube educational surgical videos, public arthroscopy datasets like SAR-RARP50 or Knee-Arthroscopy-3D).
- **Clinical Rationale:** Arthroscopy is the clinical "gold standard" ground truth for visualizing ACL/MCL tears, meniscal fraying, and cartilage ulceration directly inside the joint capsule.
- **Potential ML Applications:**
  1. *Anatomical Priors:* Train an anatomical feature extractor or self-supervised backbone on joint structures.
  2. *Educational & Sanity Checking:* Use video keyframes to cross-reference ambiguous tear geometries and tear patterns (radial, longitudinal, horizontal cleavage, bucket handle).
  3. *Multimodal Pretraining:* If open-access paired MRI-arthroscopy data exists (e.g., in medical research papers), use it to fine-tune MRI embeddings.
- **Rules Compliance Check:** Any external video/image datasets used for training must be publicly and freely accessible and posted in the official Kaggle external data thread before the deadline.

---

## 2. DICOM Metadata Fingerprinting as Auxiliary Features
- Train an auxiliary LightGBM / CatBoost model on DICOM header attributes (scanner manufacturer, magnetic field strength, slice thickness, echo times) to calibrate prior probabilities per institution.

---

## 3. Anatomical Plane-Specific Expert Routing (Mixture of Experts)
- Route Sagittal series specifically to ACL/PCL/Meniscal heads.
- Route Coronal series specifically to MCL/LCL/Collateral heads.
- Route Axial series specifically to Patellofemoral (PF OA) & Effusion heads.

---

## 4. True 3D Morphometrics, Digital Meshes & 3D Printing Workflow
- **Idea:** Reconstruct 3D surface meshes / point clouds from segmented bone and cartilage surfaces (femur, tibia, patella) to extract physical 3D biomechanical features and enable physical visualization.
- **Biomarkers & Quantifications (Digital Analysis):**
  1. *3D Joint Space Width (JSW):* Minimum surface-to-surface Euclidean distance maps between femoral condyles and tibial plateau (direct quantification of Medial/Lateral OA).
  2. *Volumetric Cartilage & Bone Spur Loss:* Measure exact cubic volume ($mm^3$) of osteophytes, cartilage thinning, and subchondral bone surface roughness.
  3. *3D Alignment Angles:* True mechanical axes, coronal tibiofemoral angle (varus/valgus alignment), and patellar tilt without 2D projection distortion.
  4. *Geometric Deep Learning:* Option to feed point clouds directly into PointNet++ / Graph Neural Networks (GNNs) or project Grad-CAM heatmaps onto 3D surfaces for error analysis.
- **Physical 3D Printing Utility:**
  - *Analytical Utility:* Plastic prints are for human tactile evaluation, surgical planning, and hackathon presentation props / video demos (holding a physical KL-0 normal vs. KL-4 arthritic joint). Algorithmic training uses the digital mesh/point cloud.
  - *Conversion Pipeline:* DICOM stack $\rightarrow$ Segmentation (bone/cartilage isolation) $\rightarrow$ Marching Cubes (isosurface mesh) $\rightarrow$ Mesh cleanup/watertight post-processing $\rightarrow$ `.stl` / `.3mf` export $\rightarrow$ Slicer.
- **Legal & Data Licensing Boundaries (Thingiverse / Public Sharing):**
  - **RSNA Competition Data Prohibition (Section 2.4.b.1):** The dataset is governed by the **RSNA MIRA License** and competition rules which state: *"You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition."*
  - **Verdict on Uploading to Thingiverse:** **Strictly prohibited for RSNA competition scans.** Derived 3D mesh files (`.stl`) cannot be uploaded to public platforms where non-participants can download them.
  - **Open-Access Alternatives for Public 3D Models:** To legally publish knee pathology 3D prints on Thingiverse/Printables, source scans from CC-BY / CC0 / Open-Access repositories such as:
    - *The Cancer Imaging Archive (TCIA)* (CC-BY collections)
    - *Open Knee (SimTK)* (open biomechanics research models)
    - *Zenodo / Figshare* open orthopedic datasets
- **Validation Plan:** Benchmark an explicit 3D morphometric feature extractor / GNN branch against the pure 2.5D deep learning baseline during the Phase 5 iteration sprint.

---

## 5. Small Vision-Language Model (VLM) Fine-Tuning: Gemma 4 vs. PaliGemma
- **Generational Evolution & Architecture:**
  - **PaliGemma / PaliGemma 2 (2024 - Composite VLM):** Earlier composite architecture combining a separate **SigLIP** vision encoder with a **Gemma** decoder via a linear projection bottleneck.
  - **Gemma 4 (2026 - Latest Generation / Native Multimodal):** The newest open-weights generation (Apache 2.0). It features **native multimodal processing** (text, vision, and audio) directly in the backbone, eliminating separate encoder bottlenecks.
- **Why Small Gemma 4 (E2B / E4B) is Ideal for Knee MRI:**
  - *Native Image Token Ingestion:* Directly accepts 2D/multi-image MRI keyframes and DICOM metadata prompts in a unified context window.
  - *Compact Footprint:* The **Effective 2B (E2B)** and **Effective 4B (E4B)** variants are fast, lightweight, and easily fit into Kaggle single-GPU kernels (T4/P100 memory & 9-hour inference timeout) with 4-bit/8-bit QLoRA.
  - *Multimodal Embeddings & Rationale:* Extract penultimate dense representations or structured diagnostic rationale tokens (e.g., grading joint space narrowing and tear severity) for downstream ensembling with CNN/ViT backbones.
- **Volumetric Adaptation Strategy:**
  - Feed 2.5D multi-slice keyframe montages or multi-image slice sequences into Gemma 4's native visual context window.
- **Exploration Phase:** Phase 5 (Model Exploration & Iteration Sprint).


