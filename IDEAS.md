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

---

## 6. Vision-to-Report Translation + JEV Pipeline (Multimodal Radiologist Pipeline)
- **Idea:** Pass multi-slice DICOM series into a compact local Multimodal Vision-Language Model (VLM) to generate detailed radiological findings / structured clinical descriptions, then feed that text directly into the calibrated Jev "System One" NLP parser (which already achieves 0.878 Macro AUC on text reports).
- **Clinical & ML Rationale:** 
  - Mirrors human radiologist workflow: $\text{DICOM Pixels} \xrightarrow{\text{Visual Inspection}} \text{Radiology Report} \xrightarrow{\text{Clinical Review}} \text{Diagnostic Scoring}$.
  - Takes maximum advantage of our proven, highly accurate Jev NLP extractor on natural language.
- **Challenges & Constraints:**
  1. *Kaggle Sandbox Lock:* Submission runs with **Internet Disabled**. No external APIs (Gemini 1.5 Pro, GPT-4o) allowed at test time. Must run an open-weights VLM (e.g., Gemma 4 E2B, PaliGemma, or Florence-2) locally on a 16GB T4 GPU.
  2. *Lossy Bottleneck & Compounding Errors:* Converting continuous pixel gradients to discrete language tokens can lose subtle visual signals (e.g. 2mm partial tears or faint subchondral edema). Total error compounds: $\text{Error}_{\text{Vision-to-Text}} + \text{Error}_{\text{Jev}}$.
  3. *Inference Throughput:* Generating long text per patient across thousands of test studies can strain the Kaggle 9-hour inference window unless heavily quantized or batched.
- **Refined Tournament Variations:**
  - *Variation A (Direct Structured JSON VLM):* Prompt the local VLM to output structured JSON findings directly (`{"ACL": 0.95, "Medial Meniscus": 0.80}`) rather than narrative prose.
  - *Variation B (BiomedCLIP Vision-Language Similarity):* Project MRI slice embeddings directly against clinical concept text embeddings (e.g. "Full thickness tear of anterior cruciate ligament") using dot-product similarity, bypassing text generation entirely.
  - *Variation C (Auxiliary Report Supervision):* Train our vision backbone (ResNet/ConvNeXt) to predict Jev targets while simultaneously predicting report text embeddings as an auxiliary loss.
- **Status:** Backlog / Alternative track. Priority remains on Fast Caching ($384 \times 384$) + Multi-backbone MoE ensembling.




## 7. Low-GPU / Rapid Notebook Explorations
- **Cleanlab / Confident Learning Label Quality Audit:** Medical ML is bottlenecked by label quality, and the Jev NLP parser makes systematic errors (e.g., 70% false alarm rate on effusion). Use out-of-fold (OOF) predictions to automatically flag the 5% most likely label errors. High leverage, no GPU required.
- **Anatomical Co-occurrence & Bayesian Post-Processing:** Use medical logic (ACL tears cause "kissing" bone contusions; severe OA destroys the meniscus) as a sanity check. Model joint conditional probabilities via a Bayesian network on model logits to boost Macro AUC without retraining.
- **OOF Target-Specific Blending:** Optimize class-specific blend weights and temperature scaling on validation predictions (e.g., CoAtNet vs. ConvNeXt on Lateral Meniscus) using Nelder-Mead or `scipy.optimize`. Free leaderboard points.
- **Frozen Feature Extraction + Fast Probing:** Cache the 768-D embeddings from the vision backbone to disk in one pass. Allows testing dozens of classification heads, MoE router variants, or loss functions in seconds on CPU.
- **DICOM Metadata Fingerprinting:** Train a fast LightGBM / CatBoost model on DICOM headers (scanner manufacturer, magnetic field strength, etc.) to learn institutional priors (Note: high risk of shortcut learning/overfitting to training distribution).

## 8. Beautiful Visualization Concepts (Notebooks)
- **The UMAP "Galaxy of Knees":** Extract 768-D embeddings, project to 2D via UMAP, and color by Jev labels. Interactive tooltips (Bokeh/Plotly) show the actual MRI slice and report on hover.
- **Interactive 3D Voxel Knee Rendering:** Use `scikit-image` marching cubes to threshold 3D MRI arrays (isolating bone/cartilage) and render an interactive 3D mesh via Plotly or PyVista.
- **Grad-CAM "Slicer" Animation HUD:** An interactive widget with a slider that scrolls through the MRI volume, dynamically overlaying Grad-CAM heatmaps to show the model's visual attention tracking structures like the ACL or meniscus.
- **Morphological "Eigen-Knees" (PCA Interpolation):** Flatten registered, central sagittal slices and run PCA. Visualize top components to create an animation interpolating from a healthy joint space to severe osteoarthritis.
- **The Clinical Fingerprint Radar Chart:** Overlay the 64-D visual embedding and 64-D textual embedding (from E11 report-supervision) on a radar chart to visualize how closely the model's visual representation aligns with the radiologist's text.

## 9. Commercial & MedTech Workflows (Stryker Sports Med Context)
Since the competition weights will be public, the moat is not the algorithm itself, but hardware-software integration and workflow capture:
- **Automated Surgical Pre-Planning (Software-to-Hardware Pipeline):** The algorithm detects the pathology (e.g., 15mm bucket-handle meniscus tear). Proprietary pre-op software translates this into a Bill of Materials, recommending specific Stryker suture anchors or ACL grafts.
- **The Smart Arthroscopy Tower (Intraoperative HUD):** The open algorithm highlights hidden cartilage damage on the pre-op MRI. During surgery, proprietary computer vision registers the 3D MRI model to the live camera feed, providing the surgeon with an augmented reality "minimap" to find the tear.
- **Post-Market Clinical Superiority (Data Moat):** Use the automated model to run retrospective analyses on thousands of post-op MRIs to mathematically prove that Stryker implants lead to better long-term joint preservation (e.g., less effusion) than competitors.
- **Patient Consent & Education (Sales Enablement):** Provide surgeons with an iPad app that uses the algorithm to turn confusing MRIs into clear, color-coded 3D visualizations, helping patients understand their injuries instantly and accelerating surgical bookings (using Stryker hardware).
