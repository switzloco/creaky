# RSNA Knee Abnormality Detection — Execution Plan

> **Author:** Claude (strategist/planner)
> **Executor:** Gemini 3.6 Flash
> **Last Updated:** 2026-08-12
> **Competition Deadline:** October 22, 2026

---

## Table of Contents

1. [Competition Overview](#1-competition-overview)
2. [Project Structure](#2-project-structure)
3. [Environment & Constraints](#3-environment--constraints)
4. [Phase 1: Label Mining Pipeline](#4-phase-1-label-mining-pipeline)
5. [Phase 2: DICOM Preprocessing](#5-phase-2-dicom-preprocessing)
6. [Phase 3: Model Training](#6-phase-3-model-training)
7. [Phase 4: Inference & Submission](#7-phase-4-inference--submission)
8. [Phase 5: Iteration & Improvement](#8-phase-5-iteration--improvement)
9. [Efficiency Track Strategy](#9-efficiency-track-strategy)
10. [Timeline](#10-timeline)
11. [Key Gotchas & Pitfalls](#11-key-gotchas--pitfalls)

---

## 1. Competition Overview

### What We're Building
A multi-label classifier that takes knee MRI DICOM images and predicts the probability of **12 abnormalities** per study.

### The 12 Targets
| # | Target | Description |
|---|--------|-------------|
| 1 | ACL | Anterior cruciate ligament tear |
| 2 | MCL | Medial collateral ligament tear |
| 3 | Medial Meniscus | Medial meniscus tear |
| 4 | Lateral Meniscus | Lateral meniscus tear |
| 5 | Medial OA | Medial compartment osteoarthritis |
| 6 | Lateral OA | Lateral compartment osteoarthritis |
| 7 | PF OA | Patellofemoral osteoarthritis |
| 8 | Effusion | Joint fluid accumulation |
| 9 | Synovitis | Inflammation of the synovial membrane |
| 10 | Baker's | Baker's cyst (popliteal cyst) |
| 11 | Contusion | Bone bruise / contusion |
| 12 | Fracture | Bone fracture |

### Data Facts
- **4,407 training studies** — each has MRI DICOM images + a radiology report
- **Only 58 studies** have expert-annotated ground-truth labels (the "gold standard")
- **4,349 studies** have reports only — labels must be extracted via NLP/LLM
- Reports are in **12 different languages** from 16 international sites
- `test.csv` does **NOT** contain reports — final model must work from images alone
- **Metric:** Macro-averaged AUC ROC across all 12 targets

### Key Insight
> **This competition is won or lost on label quality.** The imaging model is important,
> but the biggest differentiator is how accurately you extract training labels from
> the radiology reports. Spend at least 40% of your effort here.

---

## 2. Project Structure

Create this exact directory structure:

```
creaky/
├── README.md                           # (exists)
├── PLAN.md                             # (this file)
├── LICENSE                             # (exists)
├── requirements.txt                    # Python dependencies for local dev
├── configs/
│   └── experiment.yaml                 # Hyperparameters, paths, model config
├── scripts/
│   ├── label_mining/
│   │   ├── __init__.py
│   │   ├── regex_labeler.py            # Step 1A: Rule-based label extraction
│   │   ├── llm_labeler.py             # Step 1B: LLM-based label extraction
│   │   ├── label_validator.py         # Step 1C: Validate vs 58 gold labels
│   │   └── assemble_labels.py         # Step 1D: Merge into final train_labels.csv
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── dicom_to_png.py            # Step 2A: Convert DICOMs to PNGs
│   │   ├── dicom_metadata.py          # Step 2B: Extract & organize DICOM metadata
│   │   └── build_study_manifest.py    # Step 2C: Create study→series→slice manifest
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model.py                   # Step 3A: Model architecture
│   │   ├── dataset.py                 # Step 3B: PyTorch Dataset
│   │   ├── train.py                   # Step 3C: Training loop
│   │   ├── losses.py                  # Step 3D: Loss functions
│   │   └── augmentations.py           # Step 3E: Data augmentation
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py                 # AUC ROC computation
│       └── kaggle_utils.py            # Path helpers for Kaggle vs local
├── notebooks/
│   ├── 00_eda.ipynb                   # Exploratory data analysis
│   ├── 01_preprocess_dicoms.ipynb     # Kaggle notebook: DICOM → PNG dataset
│   ├── 02_label_mining.ipynb          # Kaggle notebook: Reports → Labels
│   ├── 03_train_model.ipynb           # Kaggle notebook: Model training
│   └── 04_inference.ipynb             # Kaggle notebook: FINAL SUBMISSION
└── tests/
    ├── __init__.py
    ├── test_regex_labeler.py           # Unit tests for regex extraction
    └── test_metrics.py                 # Unit tests for AUC computation
```

---

## 3. Environment & Constraints

### Local Development (User's Machine)
- **OS:** Windows
- **Python:** Managed by `uv` — always use `uv run python` (never bare `python`)
- **Shell:** cmd.exe only (no PowerShell)
- **No admin rights** — local installs only (`npm install`, `uv pip install`)
- **Node.js:** v24.14.0, npm 11.9.0 (on PATH)
- **No proprietary data** — only Kaggle competition data

### Kaggle Notebook Constraints
- **GPU runtime:** ≤ 9 hours
- **Internet:** DISABLED during submission
- **Disk:** ~70 GB available during session, ~20 GB for saved output
- **RAM:** ~30 GB (GPU notebooks) or ~16 GB (CPU)
- **GPU:** T4 (16 GB VRAM) or P100 (16 GB) — Kaggle Pro may get dual T4
- **Allowed:** Pre-trained models, public external data
- **Output:** Must produce `submission.csv`

### Python Dependencies (requirements.txt)
```
torch>=2.0
torchvision>=0.15
timm>=0.9                  # PyTorch Image Models (pretrained backbones)
pydicom>=2.4               # DICOM reading (dev/testing)
dicomsdl>=0.4              # Fast DICOM reading (inference)
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
albumentations>=1.3        # Image augmentation
opencv-python-headless>=4.8
pyyaml>=6.0
tqdm>=4.65
matplotlib>=3.7            # EDA/visualization
```

---

## 4. Phase 1: Label Mining Pipeline

> **Goal:** Convert 4,349 unstructured radiology reports (12 languages) into structured
> binary labels for 12 findings. Validate against 58 gold-standard labels.

### Step 1A: Regex/Lexicon Labeler

**File:** `scripts/label_mining/regex_labeler.py`

Build a rule-based system. For each of the 12 findings, create keyword sets:

```python
# Example structure — executor should expand these significantly
FINDING_PATTERNS = {
    "ACL": {
        "positive": [
            r"ACL\s+(tear|rupture|torn|disrupted|deficient|injured)",
            r"anterior cruciate\s+(tear|rupture|torn|disrupted|injury)",
            r"complete\s+ACL\s+(tear|rupture)",
            r"partial\s+ACL\s+(tear|rupture)",
        ],
        "negative": [
            r"ACL\s+(intact|normal|unremarkable)",
            r"intact\s+ACL",
            r"no\s+ACL\s+(tear|injury|abnormality)",
            r"anterior cruciate\s+(intact|normal)",
        ],
    },
    # ... repeat for all 12 findings
}
```

**Key requirements:**
1. Case-insensitive matching
2. Handle negation: "no evidence of ACL tear" → `ACL = 0`
3. Handle double negation: "cannot exclude ACL tear" → `ACL = UNK`
4. Output a DataFrame with columns: `StudyInstanceUID` + 12 findings
5. Each cell = `1.0` (positive), `0.0` (negative), or `NaN` (unknown/silent)
6. This will ONLY work on English reports — that's expected. Non-English goes to LLM.

**Negation detection strategy:**
- Check for negation words within a 5-word window before the finding keyword
- Negation words: "no", "not", "without", "absent", "negative", "unremarkable", "intact", "normal", "deny", "denies"
- Hedging words that should map to UNK: "cannot exclude", "possible", "questionable", "equivocal", "indeterminate"

### Step 1B: LLM-Based Labeler

**File:** `scripts/label_mining/llm_labeler.py`

Use an LLM to extract labels from reports where regex fails or for non-English reports.

**Prompt template:**
```
You are a radiology report parser. Given the following knee MRI radiology report,
extract whether each of the following 12 findings is present.

For each finding, respond with exactly one of:
- "YES" — the finding is clearly described as present
- "NO" — the finding is clearly described as absent or the structure is described as normal
- "UNK" — the report does not mention this finding, or the language is ambiguous

Report:
{report_text}

Respond in this exact JSON format:
{
  "ACL": "YES|NO|UNK",
  "MCL": "YES|NO|UNK",
  "Medial Meniscus": "YES|NO|UNK",
  "Lateral Meniscus": "YES|NO|UNK",
  "Medial OA": "YES|NO|UNK",
  "Lateral OA": "YES|NO|UNK",
  "PF OA": "YES|NO|UNK",
  "Effusion": "YES|NO|UNK",
  "Synovitis": "YES|NO|UNK",
  "Bakers": "YES|NO|UNK",
  "Contusion": "YES|NO|UNK",
  "Fracture": "YES|NO|UNK"
}
```

**Implementation notes:**
1. Use Gemini API (user has access) or fall back to a locally-run model on Kaggle GPU
2. Process ALL 4,349 unlabeled reports through the LLM
3. Parse the JSON response; if parsing fails, retry up to 3 times with temperature=0
4. Also run against the 58 labeled reports for validation
5. Save outputs as `llm_labels_raw.csv`
6. Consider batching reports to reduce API costs

**Important rule compliance note:**
> The competition rules require that any external LLM/API used must comply with their
> data security requirements. Enterprise/API configurations with no data retention are
> recommended. The radiology reports are de-identified, but check the rules carefully.
> Open-weights models run locally on Kaggle are the safest option.

### Step 1C: Label Validator

**File:** `scripts/label_mining/label_validator.py`

Compare your extracted labels against the 58 gold-standard expert labels.

**Metrics to compute (per finding and overall):**
- Accuracy
- Sensitivity (recall) — critical: did we catch the positives?
- Specificity — did we correctly identify negatives?
- F1 score
- Cohen's Kappa (inter-rater agreement)

**Output:** A formatted report showing per-finding performance, plus confusion matrices.

**Target:** ≥85% accuracy against gold labels before proceeding to image training.

### Step 1D: Label Assembly

**File:** `scripts/label_mining/assemble_labels.py`

Merge regex and LLM labels into final training labels:

**Merge strategy (priority order):**
1. If the study has a gold-standard label → use it (58 studies)
2. If regex and LLM agree → use the agreed label
3. If they disagree → use the LLM label (it handles nuance better)
4. If both are UNK → assign `0.5` (soft label expressing uncertainty)
5. Map YES → `1.0`, NO → `0.0`, UNK → `0.5`

**Output:** `train_labels.csv` with columns:
```
StudyInstanceUID, ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Bakers, Contusion, Fracture,
label_source (gold|regex|llm|soft)
```

---

## 5. Phase 2: DICOM Preprocessing

> **Goal:** Convert raw DICOM images (~500 GB) into an efficient format (~30-50 GB)
> that can be quickly loaded during training. This runs as a Kaggle notebook that
> produces a Kaggle Dataset.

### Step 2A: DICOM Metadata Extraction

**File:** `scripts/preprocessing/dicom_metadata.py`

For each DICOM file, extract and save:
```python
metadata_fields = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",       # unique slice ID
    "InstanceNumber",       # slice ordering hint
    "ImageOrientationPatient",  # 6 direction cosines
    "ImagePositionPatient",     # 3D position (x,y,z)
    "PixelSpacing",
    "SliceThickness",
    "Rows", "Columns",
    "SeriesDescription",    # e.g. "SAG T2", "COR PD FS"
    "MagneticFieldStrength",
    "Manufacturer",
    "WindowCenter", "WindowWidth",
    "RescaleIntercept", "RescaleSlope",
    "PhotometricInterpretation",
    "BitsAllocated",
]
```

**Output:** `study_metadata.parquet` — one row per DICOM slice

### Step 2B: Study Manifest

**File:** `scripts/preprocessing/build_study_manifest.py`

Group slices into a hierarchical manifest:

```python
# Output structure:
{
    "study_uid_1": {
        "series": [
            {
                "series_uid": "...",
                "description": "SAG T2 FS",
                "plane": "sagittal",        # inferred from ImageOrientationPatient
                "num_slices": 30,
                "slice_uids_ordered": [...], # sorted by spatial position
            },
            # ... more series
        ]
    }
}
```

**Plane detection logic:**
- Extract the `ImageOrientationPatient` (6 floats: row_x, row_y, row_z, col_x, col_y, col_z)
- Compute the normal vector: `normal = cross(row_vec, col_vec)`
- The dominant component of `normal` determines the plane:
  - If `|normal_x|` is largest → **sagittal**
  - If `|normal_y|` is largest → **coronal**
  - If `|normal_z|` is largest → **axial**

**Slice ordering logic:**
- Compute `position_along_normal = dot(ImagePositionPatient, normal)`
- Sort slices by this value

### Step 2C: DICOM to PNG Conversion

**File:** `scripts/preprocessing/dicom_to_png.py`

This is the heavy-lifting step. Run as Kaggle notebook `01_preprocess_dicoms.ipynb`.

**For each DICOM slice:**
1. Read pixel data using `dicomsdl` (fast) with fallback to `pydicom`
2. Apply rescale: `pixel = pixel * RescaleSlope + RescaleIntercept`
3. Apply windowing (VOI LUT):
   ```python
   def apply_window(pixel_array, window_center, window_width):
       lower = window_center - window_width / 2
       upper = window_center + window_width / 2
       pixel_array = np.clip(pixel_array, lower, upper)
       pixel_array = ((pixel_array - lower) / (upper - lower) * 255).astype(np.uint8)
       return pixel_array
   ```
   - If no window info in DICOM, use 1st/99th percentile of pixel values
4. Resize to **384×384** (good balance of resolution vs memory)
5. Save as PNG: `{study_uid}/{series_uid}/{slice_index:04d}.png`

**Output:** Save as a Kaggle Dataset. Expect ~30-50 GB compressed.

### Step 2D: Laterality Detection

Determine if the MRI is of a left or right knee:
- Use the x-coordinate of `ImagePositionPatient` for sagittal images
- Positive x → patient's left side, Negative x → patient's right side
- Add a `laterality` column to the manifest (useful for augmentation decisions)

---

## 6. Phase 3: Model Training

### Step 3A: Model Architecture

**File:** `scripts/training/model.py`

**Architecture: 2.5D CNN with Attention Pooling**

```python
import torch
import torch.nn as nn
import timm

class SliceEncoder(nn.Module):
    """Encodes individual 2D slices using a pretrained backbone."""
    def __init__(self, backbone_name="efficientnet_b3", pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,          # 2.5D: center slice + neighbors
            num_classes=0,       # remove classification head
            global_pool="avg",
        )
        self.feature_dim = self.backbone.num_features

    def forward(self, x):
        # x: (batch, 3, H, W) — 3 adjacent slices as RGB channels
        return self.backbone(x)  # (batch, feature_dim)


class AttentionPool(nn.Module):
    """Attention-weighted pooling over a variable number of slice features."""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask=None):
        # features: (batch, num_slices, feature_dim)
        # mask: (batch, num_slices) — True for valid slices
        attn_weights = self.attention(features).squeeze(-1)  # (batch, num_slices)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = torch.einsum("bn,bnd->bd", attn_weights, features)
        return pooled  # (batch, feature_dim)


class KneeAbnormalityModel(nn.Module):
    """Full model: per-series encoding → study-level classification."""
    def __init__(self, backbone_name="efficientnet_b3", num_targets=12, max_series=6):
        super().__init__()
        self.slice_encoder = SliceEncoder(backbone_name)
        self.attention_pool = AttentionPool(self.slice_encoder.feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.slice_encoder.feature_dim * max_series, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_targets),
        )
        self.max_series = max_series

    def forward(self, series_slices, series_masks):
        """
        series_slices: list of (batch, num_slices, 3, H, W) — one per series
        series_masks: list of (batch, num_slices) — one per series
        """
        series_embeddings = []
        for slices, mask in zip(series_slices, series_masks):
            B, N, C, H, W = slices.shape
            # Encode all slices
            flat = slices.view(B * N, C, H, W)
            features = self.slice_encoder(flat)  # (B*N, D)
            features = features.view(B, N, -1)   # (B, N, D)
            # Attention pool
            pooled = self.attention_pool(features, mask)  # (B, D)
            series_embeddings.append(pooled)

        # Pad to max_series if fewer series
        D = series_embeddings[0].shape[-1]
        while len(series_embeddings) < self.max_series:
            series_embeddings.append(torch.zeros_like(series_embeddings[0]))

        # Concat and classify
        study_features = torch.cat(series_embeddings[:self.max_series], dim=-1)
        logits = self.classifier(study_features)  # (B, 12)
        return logits
```

**Why this architecture:**
- **2.5D (3 adjacent slices as RGB):** Gives spatial context without 3D conv cost
- **Attention pooling:** Different slices matter for different findings — ACL is best seen on specific sagittal slices, effusion on axial, etc.
- **Per-series processing:** Each MRI series (sagittal, coronal, axial) sees different anatomy. Process them separately, then fuse.
- **EfficientNet-B3:** Good accuracy/speed tradeoff. Can upgrade to B4 or ConvNeXt-Small if GPU budget allows.

### Step 3B: Dataset

**File:** `scripts/training/dataset.py`

```python
class KneeStudyDataset(torch.utils.data.Dataset):
    """
    Loads a knee MRI study as a set of series, each containing ordered slices.

    For 2.5D: each "slice" is actually 3 adjacent slices stacked as RGB channels.
    """

    def __init__(self, manifest, labels_df, image_dir, transform=None,
                 max_slices_per_series=32, max_series=6):
        self.studies = list(manifest.keys())
        self.manifest = manifest
        self.labels = labels_df.set_index("StudyInstanceUID")
        self.image_dir = image_dir
        self.transform = transform
        self.max_slices = max_slices_per_series
        self.max_series = max_series
        self.target_cols = [
            "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
            "Medial OA", "Lateral OA", "PF OA", "Effusion",
            "Synovitis", "Bakers", "Contusion", "Fracture"
        ]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study_uid = self.studies[idx]
        study_info = self.manifest[study_uid]
        labels = self.labels.loc[study_uid, self.target_cols].values.astype(np.float32)

        all_series_slices = []
        all_series_masks = []

        for series_info in study_info["series"][:self.max_series]:
            slice_paths = series_info["slice_uids_ordered"]
            # Sample or pad to max_slices
            slices = self._load_series_slices(study_uid, series_info, slice_paths)
            mask = torch.ones(len(slices), dtype=torch.bool)

            # Pad if needed
            if len(slices) < self.max_slices:
                pad_count = self.max_slices - len(slices)
                slices = torch.cat([slices, torch.zeros(pad_count, 3, 384, 384)])
                mask = torch.cat([mask, torch.zeros(pad_count, dtype=torch.bool)])

            all_series_slices.append(slices[:self.max_slices])
            all_series_masks.append(mask[:self.max_slices])

        return {
            "series_slices": all_series_slices,
            "series_masks": all_series_masks,
            "targets": torch.tensor(labels),
            "study_uid": study_uid,
        }

    def _load_series_slices(self, study_uid, series_info, slice_paths):
        """Load slices and create 2.5D stacks (3 adjacent slices as channels)."""
        images = []
        for path in slice_paths:
            img = cv2.imread(str(self.image_dir / study_uid / series_info["series_uid"] / path),
                           cv2.IMREAD_GRAYSCALE)
            images.append(img)

        stacks = []
        for i in range(len(images)):
            prev_img = images[max(0, i-1)]
            curr_img = images[i]
            next_img = images[min(len(images)-1, i+1)]
            stack = np.stack([prev_img, curr_img, next_img], axis=0)  # (3, H, W)
            if self.transform:
                stack = self.transform(stack)
            stacks.append(torch.tensor(stack, dtype=torch.float32) / 255.0)

        return torch.stack(stacks) if stacks else torch.zeros(1, 3, 384, 384)
```

### Step 3C: Training Loop

**File:** `scripts/training/train.py`

**Key training decisions:**

| Parameter | Value | Rationale |
|---|---|---|
| Folds | 5-fold stratified | Stratify by `label_source` and label prevalence |
| Backbone | EfficientNet-B3 | Good speed/accuracy tradeoff |
| Optimizer | AdamW | Standard for medical imaging |
| Learning rate | 1e-4 (backbone), 1e-3 (head) | Differential LR for pretrained vs new layers |
| Scheduler | Cosine annealing with warmup (5 epochs) | Smooth convergence |
| Loss | BCE with label smoothing (α=0.05) | Handles noisy labels from NLP |
| Mixed precision | fp16 via `torch.cuda.amp` | 2x throughput on T4 |
| Epochs | 15-20 | With early stopping on val AUC (patience=5) |
| Batch size | 4 studies (limited by VRAM) | Accumulate gradients over 4 steps for effective BS=16 |
| Augmentations | Horizontal flip, random rotation ±15°, brightness/contrast, elastic deform | Standard medical imaging augmentations |
| Sampling | Oversample rare findings (Fracture, Baker's) | Balance class distribution |

**Stratified K-Fold strategy:**
- Use `MultilabelStratifiedKFold` from `iterstrat` package
- This ensures each fold has similar label distributions across all 12 targets
- Critical because some findings (Fracture, Baker's) may be rare

**Training outputs:**
- Save best model weights per fold: `fold_{i}_best.pth`
- Save training logs: `fold_{i}_log.csv`
- Save OOF (out-of-fold) predictions for ensemble calibration

### Step 3D: Loss Function

**File:** `scripts/training/losses.py`

```python
class SoftBCEWithLogitsLoss(nn.Module):
    """
    BCE loss that handles soft labels (0.5 for UNK).
    Optionally applies label smoothing for noisy NLP-derived labels.
    """
    def __init__(self, label_smoothing=0.05, unk_weight=0.3):
        super().__init__()
        self.smoothing = label_smoothing
        self.unk_weight = unk_weight  # downweight UNK samples

    def forward(self, logits, targets):
        # Apply label smoothing
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        # Create weight mask: lower weight for soft-labeled (UNK=0.5) samples
        weights = torch.ones_like(targets)
        unk_mask = (targets == 0.5)
        weights[unk_mask] = self.unk_weight

        loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, weight=weights, reduction="mean"
        )
        return loss
```

### Step 3E: Augmentations

**File:** `scripts/training/augmentations.py`

Use `albumentations` library:
```python
import albumentations as A

def get_train_transforms(image_size=384):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.2),
        A.ElasticTransform(alpha=50, sigma=10, p=0.1),
    ])

def get_val_transforms(image_size=384):
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
```

**Note on horizontal flip:** Only apply if you've normalized laterality in preprocessing.
A left knee flipped looks like a right knee. If your model isn't laterality-aware, this is fine.
If it is, be careful.

---

## 7. Phase 4: Inference & Submission

> **This is the notebook that gets submitted to Kaggle. It must run in ≤9 hours
> with NO internet access.**

### File: `notebooks/04_inference.ipynb`

**Pipeline:**
1. Load model weights from attached Kaggle Dataset (all 5 folds)
2. Read test DICOM paths from `/kaggle/input/rsna-knee-abnormality-detection/test_dicom/`
3. For each test study:
   a. Read DICOM metadata → determine series/planes/ordering
   b. Apply same preprocessing (window, rescale, resize to 384×384)
   c. Create 2.5D stacks
   d. Run through all 5 fold models
   e. Average predictions across folds
4. Output `submission.csv`:
   ```
   StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
   ```

**Speed optimizations for inference:**
```python
# Use these for faster inference:
model.eval()
model.half()                                    # fp16 inference
torch.backends.cudnn.benchmark = True           # auto-tune convolutions
torch.set_grad_enabled(False)                   # disable gradient tracking

# Or use torch.compile for 10-30% speedup (PyTorch 2.0+):
model = torch.compile(model, mode="reduce-overhead")
```

**Use `dicomsdl` for fast DICOM reads:**
```python
import dicomsdl
dcm = dicomsdl.open(filepath)
pixel_array = dcm.pixelData()
# 5-10x faster than pydicom for batch reads
```

---

## 8. Phase 5: Iteration & Improvement

After getting a baseline submission, iterate on these levers (in order of expected impact):

### High Impact
1. **Improve label quality** — Try different LLM prompts, use chain-of-thought, add language-specific regex patterns
2. **Add more backbones** — EfficientNet-B4, ConvNeXt-Small, DINOv2 → ensemble for diversity
3. **Series selection** — Instead of using all series, identify which plane is most informative for each finding (e.g., sagittal for ACL, axial for PF OA)

### Medium Impact
4. **Test-time augmentation (TTA)** — Horizontal flip + average predictions
5. **Pseudo-labeling** — Use confident predictions on unlabeled data to retrain
6. **Stacking/blending** — Train a lightweight model on top of OOF predictions

### Lower Impact (but easy wins)
7. **Threshold optimization** — Tune per-finding thresholds on validation set (though AUC is threshold-independent, this helps for tie-breaking)
8. **Metadata features** — Scanner manufacturer, field strength, institution as auxiliary features
9. **Post-processing** — Correlate findings (e.g., contusion + fracture often co-occur)

---

## 9. Efficiency Track Strategy

The efficiency score formula:
$$\text{Efficiency} = \frac{T}{\left(\frac{S - S_{\text{baseline}}}{S_{\max} - S_{\text{baseline}}}\right)^2}$$

Where $T$ = runtime in seconds, $S$ = your AUC score.

**Key insight:** The denominator is *squared*, so a small improvement in AUC helps much more than the same proportional reduction in runtime.

**Strategy for efficiency track:**
| Optimization | Expected Impact |
|---|---|
| Single best fold instead of 5-fold ensemble | 5x faster, ~0.5% AUC loss |
| `dicomsdl` instead of `pydicom` | 5-10x faster DICOM reads |
| fp16 inference | ~2x GPU throughput |
| `torch.compile()` | 10-30% inference speedup |
| Smaller backbone (EfficientNet-B0) | 3x faster, ~1-2% AUC loss |
| Reduce image size (256×256) | 2x faster, ~0.5% AUC loss |
| Skip unnecessary series | Variable, up to 2-3x faster |

**Recommendation:** Submit one "max accuracy" run (5-fold, B3, 384px) for main leaderboard,
and one "efficiency" run (1-fold, B0, 256px, compiled) for efficiency track.

---

## 10. Timeline

| Week | Dates | Focus | Deliverable |
|---|---|---|---|
| 1 | Aug 12-18 | EDA + Regex labeler | `regex_labeler.py`, `00_eda.ipynb` |
| 2 | Aug 19-25 | LLM labeler + validation | `llm_labeler.py`, `label_validator.py`, `train_labels.csv` |
| 3 | Aug 26 - Sep 1 | DICOM preprocessing | `01_preprocess_dicoms.ipynb` → Kaggle Dataset |
| 4 | Sep 2-8 | Model architecture + dataset | `model.py`, `dataset.py`, local smoke tests |
| 5 | Sep 9-15 | First training run on Kaggle | `03_train_model.ipynb`, first fold trained |
| 6 | Sep 16-22 | Full 5-fold training | All fold weights saved as Kaggle Dataset |
| 7 | Sep 23-29 | First submission + debug | `04_inference.ipynb`, first leaderboard score |
| 8 | Oct 1-7 | Label improvement + retrain | Improved labels, retrained models |
| 9 | Oct 8-14 | Ensembling + efficiency track | Multi-backbone ensemble, efficiency submission |
| 10 | Oct 15-22 | Final tuning + submission | Final submissions selected |

---

## 11. Key Gotchas & Pitfalls

### Data Pitfalls
- ⚠️ **Unlabeled ≠ Negative:** Empty cells in `train.csv` mean "no gold label", NOT "finding absent". This is the #1 mistake new participants make.
- ⚠️ **Report language:** Reports are in 12 languages. English-only regex will miss ~40% of studies.
- ⚠️ **DICOM slice ordering:** Filenames are random UUIDs. You MUST sort by spatial position using `ImageOrientationPatient` and `ImagePositionPatient`.
- ⚠️ **Windowing:** Different MRI sequences need different windowing. Use the DICOM header values when available.

### Training Pitfalls
- ⚠️ **Data leakage:** Never use report text at inference time — `test.csv` has no reports.
- ⚠️ **Class imbalance:** Some findings (e.g., Fracture) may be very rare. Use weighted sampling or focal loss.
- ⚠️ **Label noise:** NLP-derived labels WILL have errors. Use label smoothing and robust losses.
- ⚠️ **Memory:** Each study has multiple series × dozens of slices. You cannot load everything into GPU memory. Process series sequentially.

### Submission Pitfalls
- ⚠️ **No internet:** Your inference notebook cannot download anything. All model weights must be attached as Kaggle Datasets.
- ⚠️ **9-hour limit:** Profile your inference speed on a sample. If one study takes 30 seconds and there are 500 test studies, that's 4+ hours just for inference.
- ⚠️ **Output format:** File must be named exactly `submission.csv` with exact column headers matching the sample submission.
- ⚠️ **Missing predictions:** If you skip any StudyInstanceUID, you'll get an error. Predict for ALL test studies.

### Ethics & Compliance
- ✅ Only use public competition data — no proprietary datasets
- ✅ De-identified patient data — no PHI concerns
- ✅ Open-source everything only if claiming a prize
- ✅ Check employer IP agreement before claiming any prize money
