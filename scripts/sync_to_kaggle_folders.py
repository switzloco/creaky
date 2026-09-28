"""Sync compiled notebooks to their respective Kaggle push directories."""
import shutil
from pathlib import Path

train_src = Path("train-notebook.ipynb")
train_dst = Path("notebooks/training-book/training-book.ipynb")
if train_src.exists():
    shutil.copyfile(train_src, train_dst)
    print(f"Copied {train_src} -> {train_dst} ({train_dst.stat().st_size:,} bytes)")

sub_src = Path("submission-notebook.ipynb")
sub_dst = Path("notebooks/submission-notebook/submission-notebook.ipynb")
if sub_src.exists():
    shutil.copyfile(sub_src, sub_dst)
    print(f"Copied {sub_src} -> {sub_dst} ({sub_dst.stat().st_size:,} bytes)")
