"""Generate standalone notebooks for Fold 1 and Fold 2 training with exact script parity."""
import json
import re
from pathlib import Path

def generate_fold_notebook(fold_num: int, out_ipynb: Path):
    script_text = Path("scripts/training/train_kaggle_notebook.py").read_text(encoding="utf-8")
    
    # Replace "fold": 0 with "fold": fold_num
    modified_text = re.sub(r'"fold":\s*\d+,', f'"fold": {fold_num},', script_text, count=1)
    
    assert f'"fold": {fold_num},' in modified_text, f"Failed to set fold {fold_num} in script"
    
    lines = [line + "\n" for line in modified_text.splitlines()]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# RSNA Knee Abnormality Detection — Multi-Planar MoE Training (Fold {fold_num})\n",
                    "\n",
                    f"E11 Report-Supervised Multi-Planar ConvNeXt-Small MoE training for Fold {fold_num}.\n"
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

    out_ipynb.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Generated {out_ipynb} for fold {fold_num} ({len(lines)} lines)")

if __name__ == "__main__":
    generate_fold_notebook(1, Path("notebooks/training-book-f1/training-book-f1.ipynb"))
    generate_fold_notebook(2, Path("notebooks/training-book-f2/training-book-f2.ipynb"))
