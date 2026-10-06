"""Generate standalone notebook for E16 / N2 (Mirror-Knee Horizontal Flip, Fold 0)."""
import json
import re
from pathlib import Path

def generate_n2_notebook():
    script_text = Path("scripts/training/train_kaggle_notebook.py").read_text(encoding="utf-8")
    
    # Configure for E16 / N2: fold 0, hflip True
    modified_text = re.sub(r'"fold":\s*\d+,', '"fold": 0,', script_text, count=1)
    modified_text = re.sub(r'"hflip":\s*False,', '"hflip": True,', modified_text, count=1)
    
    assert '"fold": 0,' in modified_text
    assert '"hflip": True,' in modified_text
    
    lines = [line + "\n" for line in modified_text.splitlines()]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# RSNA Knee Abnormality Detection — Experiment E16 (Novel Track N2: Mirror-Knee Flip)\n",
                    "\n",
                    "Ablation on Fold 0: ConvNeXt-Small MoE + E11 Report Supervision + Volume-Consistent Horizontal Flip (p=0.5).\n",
                    "Testing the hypothesis that anatomical mirror symmetry across Left & Right knees yields a non-destructive 2x augmentation.\n"
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

    out_p = Path("notebooks/training-book-n2/training-book-n2.ipynb")
    out_p.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Generated {out_p} ({len(lines)} lines)")

if __name__ == "__main__":
    generate_n2_notebook()
