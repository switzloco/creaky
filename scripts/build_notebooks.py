"""Convert python training and submission scripts into standard standalone Kaggle Jupyter notebooks."""
import json
from pathlib import Path

def script_to_ipynb(script_path: str, ipynb_path: str, title: str, description: str):
    script_text = Path(script_path).read_text(encoding="utf-8")

    # Format code lines as Jupyter cell source
    lines = [line + "\n" for line in script_text.splitlines()]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    f"{description}\n"
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

    out_file = Path(ipynb_path)
    out_file.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Generated notebook: {ipynb_path} ({len(lines)} code lines)")


if __name__ == "__main__":
    script_to_ipynb(
        script_path="scripts/training/train_kaggle_notebook.py",
        ipynb_path="train-notebook.ipynb",
        title="RSNA Knee Abnormality Detection — Multi-Planar MoE Training",
        description="Trains KneeAnatomicalMoEClassifier (ResNet34) on RSNA Knee Abnormality dataset using calibrated Jev 0.878 AUC soft labels and confidence weighting."
    )

    script_to_ipynb(
        script_path="scripts/submission/submission_notebook.py",
        ipynb_path="submission-notebook.ipynb",
        title="RSNA Knee Abnormality Detection — Inference & Submission",
        description="Self-contained inference notebook for Kaggle code submission. Loads trained MoE checkpoint, preprocesses multi-planar DICOM series, and outputs submission.csv."
    )

    script_to_ipynb(
        script_path="scripts/preprocessing/cache_kaggle_kernel.py",
        ipynb_path="caching-notebook.ipynb",
        title="RSNA Knee Abnormality Detection — Fast Slice Pre-Caching",
        description="Extracts, window-normalizes, and packs MRI slices across Sagittal, Coronal, and Axial planes into compact .npz arrays."
    )

