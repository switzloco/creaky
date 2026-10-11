import json
import re
from pathlib import Path

def generate_n4_fold(fold_num: int):
    source_nb_path = Path("notebooks/training-book-n4/training-book-n4.ipynb")
    nb = json.loads(source_nb_path.read_text(encoding="utf-8"))
    
    # Update markdown cell
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            cell["source"] = [
                f"# RSNA Knee Abnormality Detection — Track N4 (The Meniscus Fix) + Cleanlab (Fold {fold_num})\n",
                "\n",
                f"Trains KneeAnatomicalMoEClassifier (ConvNeXt-Small) on Track N4 cached slices for Fold {fold_num}:\n",
                "- Smart Sequence Prioritization: 94.2% Fluid-Sensitive (T2/PD FS) Sagittal & 96.4% Coronal slices.\n",
                "- Wide Depth Sampling: 2% to 98% volume depth (zero clipping of peripheral lateral meniscus).\n",
                "- Cleanlab Confident Learning: 2,634 noisy Silver labels zero-weighted (Abs_Diff >= 0.50).\n",
                "- Report Supervision: 64-D SVD radiology report fingerprint auxiliary head.\n"
            ]
            break
            
    # Update code cell to set fold number
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"]
            new_lines = []
            for line in lines:
                if '"fold":' in line:
                    new_lines.append(f'    "fold": {fold_num},          # Fold index to train (0..4)\n')
                elif 'Outputs best_model_fold_0.pt' in line:
                    new_lines.append(f'Outputs best_model_fold_{fold_num}.pt to /kaggle/working/\n')
                elif 'best_model_fold_0.pt' in line:
                    new_lines.append(line.replace('best_model_fold_0.pt', f'best_model_fold_{fold_num}.pt').replace('val_preds_fold_0.csv', f'val_preds_fold_{fold_num}.csv'))
                else:
                    new_lines.append(line)
            cell["source"] = new_lines
            break

    out_dir = Path(f"notebooks/training-book-n4-f{fold_num}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_ipynb = out_dir / f"training-book-n4-f{fold_num}.ipynb"
    out_ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Generated {out_ipynb}")
    
    meta = {
      "id": f"nswitzer/training-book-n4-cleanlab-fold-{fold_num}",
      "title": f"Training Book N4 Cleanlab (Fold {fold_num})",
      "code_file": f"training-book-n4-f{fold_num}.ipynb",
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
    generate_n4_fold(1)
    generate_n4_fold(2)
