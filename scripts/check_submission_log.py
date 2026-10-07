import io
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

slug = sys.argv[1] if len(sys.argv) > 1 else "nswitzer/creaky-0-94-ensemble"
logs = api.kernels_logs(slug)
text = ""
if isinstance(logs, dict):
    text = logs.get("log", str(logs))
elif isinstance(logs, list):
    text = "\n".join(str(x) for x in logs)
else:
    text = str(logs) if logs is not None else ""

print("--- Submission Notebook Execution Log ---")
for line in text.splitlines():
    print(line)
