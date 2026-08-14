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

## 4. True 3D Morphometrics & Joint Space Biomarkers (Mesh / Point Cloud Analysis)
- **Idea:** Reconstruct 3D surface meshes / point clouds from segmented bone and cartilage surfaces (femur, tibia, patella) to extract physical 3D biomechanical features.
- **Biomarkers & Quantifications:**
  1. *3D Joint Space Width (JSW):* Minimum surface-to-surface euclidean distance maps between femoral condyles and tibial plateau (direct quantification of Medial/Lateral OA).
  2. *Volumetric Cartilage & Bone Spur Loss:* Measure exact cubic volume ($mm^3$) of osteophytes, cartilage thinning, and subchondral bone surface roughness.
  3. *3D Alignment Angles:* True mechanical axes, coronal tibiofemoral angle (varus/valgus alignment), and patellar tilt without 2D projection distortion.
- **Validation Plan:** Benchmark an explicit 3D morphometric feature extractor / GNN branch against the pure 2.5D deep learning baseline during the Phase 5 iteration sprint.

