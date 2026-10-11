import os
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernel = "nswitzer/training-book-n4-cleanlab-fold-0"
status = api.kernels_status(kernel)
status_str = getattr(status, "status", str(status))
failure = getattr(status, "failure_message", None) or getattr(status, "failureMessage", None)
print(f"Kernel [{kernel}] Status: {status_str} | Failure: {failure}")

try:
    logs = api.kernels_logs(kernel)
    text = ""
    if isinstance(logs, dict):
        text = logs.get("log", str(logs))
    elif isinstance(logs, list):
        text = "\n".join(str(x) for x in logs)
    else:
        text = str(logs) if logs is not None else ""

    print(f"Logs fetched: {len(text)} characters")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    print(f"Total non-empty lines: {len(lines)}")
    print("\n--- Last 35 lines of log ---")
    for l in lines[-35:]:
        print(l)
except Exception as e:
    print("Error fetching logs:", e)
