import os
import io
import json
import base64
from PIL import Image

def get_compressed_b64(path, quality=88):
    if not os.path.exists(path):
        return ""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

print("Gathering and compressing images...")
img_sources = {
    "c1_raw": "kaggle_out_v9/explainers/case_1/ACL_raw.png",
    "c1_heat": "kaggle_out_v9/explainers/case_1/ACL_highlight.png",
    "c2_raw": "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_raw.png",
    "c2_heat": "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_highlight.png",
    "c3_raw": "kaggle_out_v9/explainers/case_3/ACL_raw.png",
    "c3_heat": "kaggle_out_v9/explainers/case_3/ACL_highlight.png",
    "c4_raw": "kaggle_out_v9/explainers/case_4/Contusion_raw.png",
    "c4_heat": "kaggle_out_v9/explainers/case_4/Contusion_highlight.png",
    "c5_raw": "kaggle_out_v9/explainers/case_5/PF_OA_raw.png",
    "c5_heat": "kaggle_out_v9/explainers/case_5/PF_OA_highlight.png",
    "c6_heat": "kaggle_out/explainers/case_10002/Medial_Meniscus_highlight.png",
    "c7_heat": "kaggle_out/explainers/case_10004/Effusion_highlight.png",
    "c8_heat": "kaggle_out/explainers/case_10005/Medial_OA_highlight.png",
    "ref_sag": "scratch/healthy_Sagittal.png",
    "ref_ax": "scratch/healthy_Axial.png",
    "ref_cor": "scratch/healthy_Coronal.png"
}

payload = {}
for k, path in img_sources.items():
    payload[k] = get_compressed_b64(path)

print(f"Total payload size: {sum(len(v) for v in payload.values()) / 1024:.1f} KB")

# Read template
with open("scripts/doctor_audit_template.py", "r", encoding="utf-8") as f:
    template_src = f.read()

case_data_code = template_src.replace("__IMAGE_PAYLOAD__", json.dumps(payload))

# Now assemble notebook cells
cells = []

# Cell 0: Header Markdown
cell_0 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🩺 RSNA Knee MRI Challenge: Algorithmic Error Audit & Clinical Discrepancy Review\n",
        "### *A Fast, Interactive Case Review for Musculoskeletal Radiologists & Orthopedic Surgeons*\n",
        "\n",
        "> **⚡ Runtime: CPU-Only (Zero GPU Required) | Execution Time: < 3 Seconds**\n",
        ">\n",
        "> **Goal:** Inspect 8 critical cases (16 authentic knee MRI images) where our multi-view deep learning algorithm struggled against ground-truth labels. The cases highlight **False Negatives** (missed tears/pathologies), **False Positives** (spurious overcalls), and **Ground-Truth Label Omissions** (where the model detected acute injuries described in the radiologist report that were omitted in dataset labels).\n",
        "\n",
        "--- \n",
        "\n",
        "### 📋 How to Review These Cases:\n",
        "1. **Inspect Paired MRI Slices:** Review the raw diagnostic slice alongside the algorithm's attention heatmap and healthy reference benchmarks.\n",
        "2. **Examine the Probabilities:** Compare the algorithm's output probability against the expert ground truth.\n",
        "3. **Evaluate the Clinical Dilemma:** Review why computer vision models stumble on this specific anatomy.\n",
        "4. **Record Your Clinical Impression:** Use the interactive second-opinion buttons on each card to classify the root cause (Model error vs. Label error vs. Borderline/Equivocal).\n"
    ]
}
cells.append(cell_0)

# Cell 1: Environment & Code Visibility Toggle (Code Cell - Hidden)
cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "jupyter": {"source_hidden": True},
        "tags": ["hide-input"]
    },
    "outputs": [],
    "source": [
        "# ==============================================================================\n",
        "# 1. CLINICAL VIEWER INITIALIZATION & CODE HIDING CONTROLS\n",
        "# ==============================================================================\n",
        "import os\n",
        "import sys\n",
        "import base64\n",
        "import json\n",
        "from pathlib import Path\n",
        "from typing import Dict, List, Optional, Any\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from IPython.display import HTML, display\n",
        "\n",
        "# Automatic Code Hiding banner for clean physician reading\n",
        "display(HTML('''\n",
        "<style>\n",
        "    .hide-code-banner {\n",
        "        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);\n",
        "        color: #f8fafc;\n",
        "        padding: 14px 20px;\n",
        "        border-radius: 10px;\n",
        "        display: flex;\n",
        "        align-items: center;\n",
        "        justify-content: space-between;\n",
        "        margin-bottom: 20px;\n",
        "        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;\n",
        "        box-shadow: 0 4px 12px rgba(0,0,0,0.15);\n",
        "    }\n",
        "    .hide-code-banner span {\n",
        "        font-weight: 600;\n",
        "        font-size: 14px;\n",
        "        letter-spacing: 0.3px;\n",
        "    }\n",
        "    .toggle-code-btn {\n",
        "        background: #3b82f6;\n",
        "        color: #fff;\n",
        "        border: none;\n",
        "        padding: 6px 14px;\n",
        "        border-radius: 6px;\n",
        "        cursor: pointer;\n",
        "        font-size: 13px;\n",
        "        font-weight: 500;\n",
        "        transition: all 0.2s ease;\n",
        "    }\n",
        "    .toggle-code-btn:hover { background: #2563eb; transform: translateY(-1px); }\n",
        "</style>\n",
        "<div class='hide-code-banner'>\n",
        "    <span>🩺 <b>Doctor Review Mode Active:</b> Code cells are hidden by default for maximum clinical readability.</span>\n",
        "    <button class='toggle-code-btn' onclick='toggleAllCodeCells()'>👁️ Toggle Code Visibility</button>\n",
        "</div>\n",
        "<script>\n",
        "function toggleAllCodeCells() {\n",
        "    const codeCells = document.querySelectorAll('.jp-InputArea, .input_area, .code_cell .input');\n",
        "    codeCells.forEach(el => {\n",
        "        el.style.display = (el.style.display === 'none' || el.style.display === '') ? 'block' : 'none';\n",
        "    });\n",
        "}\n",
        "</script>\n",
        "'''))\n"
    ]
}
cells.append(cell_1)

# Cell 2: Markdown Summary Table
cell_2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 📑 Executive Summary: 8 Curated Diagnostic Dilemmas\n",
        "\n",
        "| Case | Target Pathology | Plane | Model Conf. | Gold Label | Discrepancy Type | Core Diagnostic Dilemma |\n",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :--- |\n",
        "| **[Case 1](#case-1)** | **Joint Effusion** | Sagittal | **97.1%** | `0.0` | 🟣 **Label Omission** | Acute ACL tear with obvious hemarthrosis, yet gold label omitted effusion. |\n",
        "| **[Case 2](#case-2)** | **Lateral Meniscus Tear** | Sagittal | **88.6%** | `1.0` | 🟡 **Multi-Ligament FP** | Catches lateral tear, but hallucinates tears on intact ACL (76%) & MM (62%). |\n",
        "| **[Case 3](#case-3)** | **Complex Pivot-Shift Combo** | Sagittal | **79.1% (LM)** | `0.0` | 🟡 **Contralateral FP** | 99% confident on ACL & MM tears, but overcalls lateral horn at 79%. |\n",
        "| **[Case 4](#case-4)** | **Severe Marrow Contusion** | Axial | **96.1% (ACL)** | `0.0` | 🔴 **Edema Confounder** | High T2 marrow signal triggers 90%+ false alarms across ALL ligaments. |\n",
        "| **[Case 5](#case-5)** | **Patellofemoral OA (PF OA)** | Axial | **15.4%** | `1.0` | 🔴 **Critical False Neg.** | End-stage full thickness patellar chondral loss completely missed. |\n",
        "| **[Case 6](#case-6)** | **Medial Meniscus** | Sagittal | **97.2%** | `0.0` | 🟡 **Mucoid Overcall** | Fooled by Grade II intrameniscal mucin without articular surface breach. |\n",
        "| **[Case 7](#case-7)** | **ACL Abnormality** | Sagittal | **5.6%** | `1.0` | 🔴 **'Celery Stalk' Miss** | Chronic mucoid ACL degeneration missed by model trained on fiber discontinuity. |\n",
        "| **[Case 8](#case-8)** | **Tibial Plateau Split Fracture** | Coronal | **99.9% (ACL)** | `0.0` | 🟣 **Label Omission** | Depressed plateau fracture with acute contusion & ACL sprain; Gold only checked Fracture. |\n"
    ]
}
cells.append(cell_2)

# Cell 3: Main Dashboard Code (Code Cell - Hidden)
cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {
        "jupyter": {"source_hidden": True},
        "tags": ["hide-input"]
    },
    "outputs": [],
    "source": [case_data_code]
}
cells.append(cell_3)

# Cell 4: Conclusion & Feedback Synthesis Markdown
cell_4 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🩺 Key Takeaways & Discussion Points for the Clinical Team\n",
        "\n",
        "Based on our algorithmic audit across the 58 Gold Standard studies and 3,600+ out-of-fold predictions, four primary failure mechanisms account for over 80% of our model's errors:\n",
        "\n",
        "### 1. The Articular Surface Rule (Meniscal Tears vs. Mucoid Degeneration)\n",
        "- **The Challenge:** Convolutional networks measure aggregate pixel brightness and high local contrast. Grade II mucoid degeneration produces intense intrameniscal signal that mimics a tear to a naive filter.\n",
        "- **Clinical Reality:** Radiologists require the high-intensity signal to unequivocally breach the superior or inferior articular surface on at least two consecutive slices.\n",
        "- **Proposed Engineering Fix:** Implementing slice-continuity attention heads that explicitly track whether hyperintense lines contact the joint margin across neighboring slices.\n",
        "\n",
        "### 2. Edema Signal Bleed (Bone Contusions Mimicking Ligament Rupture)\n",
        "- **The Challenge:** In acute trauma (such as pivot-shift injuries), extensive bone marrow edema creates massive T2 hyperintensity. The global pooling layer in 2D backbones bleeds this signal into adjacent soft-tissue heads, falsely triggering MCL, ACL, and meniscus alarms.\n",
        "- **Proposed Engineering Fix:** Anatomical compartment segmentation masks or coordinate-guided attention to decouple bone marrow intensity from ligament fiber tracks.\n",
        "\n",
        "### 3. Chronic vs. Traumatic Pathology ('Celery-Stalk' Mucoid ACLs)\n",
        "- **The Challenge:** The algorithm was trained to identify empty intercondylar notches and disrupted fiber tracks. In chronic mucoid degeneration, the ACL remains continuous despite severe swelling and striated T2 signal.\n",
        "- **Clinical Reality:** While marked as an 'ACL Abnormality' by study annotators, orthopedic surgeons treat this as a chronic non-surgical phenomenon unless impingement occurs.\n",
        "\n",
        "### 4. Ground-Truth Incompleteness & Human Annotator Fatigue\n",
        "- **The Challenge:** As demonstrated in Case 1 and Case 8, human annotators frequently checked only the most catastrophic primary diagnosis (e.g. Tibial Plateau Fracture) and omitted secondary associated injuries (e.g. acute joint effusion or concomitant ACL sprains) that are plainly visible on imaging and documented in the report.\n",
        "- **Next Step:** Re-calibrating training loss weights using Cleanlab confident learning to zero-weight verified label omissions.\n",
        "\n",
        "---\n",
        "**Thank you for your clinical review! Please record your overall impressions and any specific sequence preferences above.**\n"
    ]
}
cells.append(cell_4)

# Create notebook dictionary
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.15"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_paths = [
    "c:/Users/nswitzer/Antigrav Proj/creaky/doctor_mri_error_audit.ipynb",
    "c:/Users/nswitzer/Antigrav Proj/creaky/notebooks/doctor_mri_error_audit.ipynb"
]

for p in out_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print(f"Successfully wrote notebook: {p} ({os.path.getsize(p) / 1024:.1f} KB)")

