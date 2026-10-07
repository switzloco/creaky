import gzip
import base64
import json
import os
from pathlib import Path

def main():
    cleanlab_csv_path = Path("data/processed/train_labels_cleanlab.csv")
    if not cleanlab_csv_path.exists():
        raise FileNotFoundError(f"{cleanlab_csv_path} does not exist.")
        
    print(f"Reading Cleanlab-filtered labels from {cleanlab_csv_path}...")
    data = cleanlab_csv_path.read_bytes()
    comp = gzip.compress(data)
    b64_str = base64.b64encode(comp).decode("ascii")
    print(f"Compressed size: {len(comp):,} bytes | Base64 string length: {len(b64_str):,} chars")
    
    src_script = Path("scripts/training/train_kaggle_notebook.py")
    content = src_script.read_text(encoding="utf-8")
    
    # 1. Replace EMBEDDED_JEV_LABELS_B64
    start_marker = "# === EMBEDDED_JEV_LABELS_START ==="
    end_marker = "# === EMBEDDED_JEV_LABELS_END ==="
    
    payload_code = f'{start_marker}\nEMBEDDED_JEV_LABELS_B64 = "{b64_str}"\n{end_marker}'
    
    if start_marker in content and end_marker in content:
        pre = content[:content.index(start_marker)]
        post = content[content.index(end_marker) + len(end_marker):]
        new_content = pre + payload_code + post
    else:
        raise ValueError("Could not find EMBEDDED_JEV_LABELS markers in source script!")
        
    # 2. Verify CONFIG: fold 0, hflip False
    # Ensure hflip is False in CONFIG
    new_content = new_content.replace('"hflip": True,', '"hflip": False,')
    new_content = new_content.replace('"fold": 1,', '"fold": 0,')
    new_content = new_content.replace('"fold": 2,', '"fold": 0,')
    
    # Add Cleanlab print marker
    marker_search = 'print(f"--> Successfully unpacked {len(out_df)} studies'
    marker_replace = 'print(f"--> [CLEANLAB FILTERED] Successfully unpacked {len(out_df)} studies (2,634 Silver label errors zero-weighted)")\n            print(f"--> Total zero-weighted target instances: {(out_df[[c for c in out_df.columns if c.endswith(\'_weight\')]] == 0.0).sum().sum()}")\n            ' + marker_search
    if marker_search in new_content:
        new_content = new_content.replace(marker_search, marker_replace, 1)
        
    out_dir = Path("notebooks/training-book-cleanlab")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    script_out = out_dir / "train_cleanlab.py"
    script_out.write_text(new_content, encoding="utf-8")
    print(f"Saved training script to: {script_out}")
    
    # 3. Build standalone Jupyter Notebook
    lines = [line + "\n" for line in new_content.splitlines()]
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RSNA Knee Abnormality Detection — Cleanlab-Filtered Training (Fold 0)\n",
                    "\n",
                    "Trains KneeAnatomicalMoEClassifier (ConvNeXt-Small) with E11 report-supervision\n",
                    "and 2,634 high-confidence Cleanlab Silver label errors zero-weighted (Abs_Diff >= 0.50).\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            }
        ],
        "metadata": {
            "kaggle": {
                "accelerator": "nvidiaTeslaT4",
                "isGpuEnabled": True,
                "isInternetEnabled": True,
                "language": "python",
                "sourceType": "notebook"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    nb_out = out_dir / "training-book-cleanlab.ipynb"
    nb_out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Generated standalone notebook: {nb_out} ({len(lines)} lines)")
    
    # 4. Write kernel-metadata.json
    metadata = {
        "id": "nswitzer/training-book-cleanlab",
        "title": "Training Book Cleanlab (Fold 0)",
        "code_file": "training-book-cleanlab.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [
            "nswitzer/creaky-caching-dataset"
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"
    }
    
    meta_out = out_dir / "kernel-metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated kernel metadata: {meta_out}")

if __name__ == "__main__":
    main()
