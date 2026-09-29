import sys
import io
import os
import shutil

# Fix Windows console charmap encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# 1. Fetch Logs
print("Fetching kernel logs...")
try:
    logs = api.kernels_logs("nswitzer/training-book")
    text = ""
    if isinstance(logs, dict):
        text = logs.get("log", str(logs))
    elif isinstance(logs, list):
        text = "\n".join(str(x) for x in logs)
    else:
        text = str(logs) if logs is not None else ""

    os.makedirs("checkpoints/convnext_trained", exist_ok=True)
    log_path = "checkpoints/convnext_trained/training.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved log ({len(text)} chars) to {log_path}")

    # Print summary lines (Epoch summaries, AUCs)
    print("\n" + "="*50)
    print("TRAINING LOG HIGHLIGHTS:")
    print("="*50)
    for line in text.splitlines():
        if any(keyword in line for keyword in ["Epoch", "Val AUC", "Train Loss", "Saved NEW BEST", "Best Validation", ">>>"]):
            # Filter out tqdm raw progress bar noise
            if not any(bar in line for bar in ["\r", "%|", "it/s"]):
                print(line.strip())
except Exception as e:
    print(f"Error fetching logs: {e}")

# 2. Download Model Checkpoint Output
print("\nDownloading kernel output files...")
output_dir = "checkpoints/convnext_trained"
try:
    api.kernels_output("nswitzer/training-book", path=output_dir)
    print(f"Output files downloaded to: {output_dir}")
    for item in os.listdir(output_dir):
        fp = os.path.join(output_dir, item)
        sz = os.path.getsize(fp) / (1024 * 1024)
        print(f"  - {item} ({sz:.2f} MB)")
except Exception as e:
    print(f"Error downloading outputs: {e}")
