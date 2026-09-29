import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

status = api.kernels_status("nswitzer/training-book")
print("Status dict / object:", getattr(status, "__dict__", status))

# Download output files safely
try:
    logs = api.kernels_logs("nswitzer/training-book")
    text = ""
    if isinstance(logs, dict):
        text = logs.get("log", str(logs))
    elif isinstance(logs, list):
        text = "\n".join(str(x) for x in logs)
    else:
        text = str(logs) if logs is not None else ""

    print(f"Logs fetched: {len(text)} characters")
    with open("checkpoints/kaggle_train_err/log.txt", "w", encoding="utf-8") as f:
        f.write(text)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    print(f"Total non-empty lines: {len(lines)}")
    print("\n--- Last 30 lines of log ---")
    for l in lines[-30:]:
        print(l)
except Exception as e:
    print("Error:", e)
