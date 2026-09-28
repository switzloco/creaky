"""Smoke test for train_kaggle_notebook.py pipeline."""
import sys
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from scripts.training.train_kaggle_notebook import CONFIG, run_training

# Set test config
CONFIG["epochs"] = 1
CONFIG["max_train_studies"] = 2
CONFIG["max_val_studies"] = 4
CONFIG["batch_size"] = 2
CONFIG["num_workers"] = 0
CONFIG["pretrained"] = False  # fast test without downloading weights


print("Running smoke test of training pipeline...")
try:
    run_training()
    print("\nTraining smoke test SUCCESS!")
except Exception as e:
    print(f"\nTraining smoke test FAILED: {e}")
    import traceback
    traceback.print_exc()
