import json
import re
from pathlib import Path

def generate_n4_training_notebook():
    source_nb_path = Path("notebooks/training-book-cleanlab/training-book-cleanlab.ipynb")
    nb = json.loads(source_nb_path.read_text(encoding="utf-8"))
    
    # Update title in markdown cell
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [
                "# RSNA Knee Abnormality Detection — Track N4 (The Meniscus Fix) + Cleanlab (Fold 0)\n",
                "\n",
                "Trains KneeAnatomicalMoEClassifier (ConvNeXt-Small) on Track N4 cached slices:\n",
                "- Smart Sequence Prioritization: 94.2% Fluid-Sensitive (T2/PD FS) Sagittal & 96.4% Coronal slices.\n",
                "- Wide Depth Sampling: 2% to 98% volume depth (zero clipping of peripheral lateral meniscus).\n",
                "- Cleanlab Confident Learning: 2,634 noisy Silver labels zero-weighted (Abs_Diff >= 0.50).\n",
                "- Report Supervision: 64-D SVD radiology report fingerprint auxiliary head.\n"
            ]
            break
            
    # Update code cell to include creaky-caching-dataset-n4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"]
            new_lines = []
            for line in lines:
                if '"/kaggle/input/creaky-caching-dataset/cached_slices",' in line:
                    new_lines.append('            "/kaggle/input/creaky-caching-dataset-n4/cached_slices",\n')
                    new_lines.append('            "/kaggle/input/creaky-caching-dataset-n4",\n')
                new_lines.append(line)
            cell["source"] = new_lines
            break

    out_dir = Path("notebooks/training-book-n4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_ipynb = out_dir / "training-book-n4.ipynb"
    out_ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {out_ipynb}")
    
    meta = {
      "id": "nswitzer/training-book-n4-cleanlab-fold-0",
      "title": "Training Book N4 Cleanlab (Fold 0)",
      "code_file": "training-book-n4.ipynb",
      "language": "python",
      "kernel_type": "notebook",
      "is_private": True,
      "enable_gpu": True,
      "enable_tpu": False,
      "enable_internet": True,
      "keywords": [],
      "dataset_sources": [],
      "kernel_sources": [
        "nswitzer/creaky-caching-dataset-n4"
      ],
      "competition_sources": [
        "rsna-knee-abnormality-detection"
      ],
      "model_sources": [],
      "machine_shape": "NvidiaTeslaT4"
    }
    
    out_meta = out_dir / "kernel-metadata.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Generated {out_meta}")

if __name__ == "__main__":
    generate_n4_training_notebook()
