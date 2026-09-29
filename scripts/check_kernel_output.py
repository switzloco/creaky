import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

status = api.kernels_status("nswitzer/training-book")
print("Status:", getattr(status, "__dict__", status))

try:
    os.makedirs("checkpoints/probe", exist_ok=True)
    out = api.kernels_output("nswitzer/training-book", "checkpoints/probe")
    print("Output files found:", os.listdir("checkpoints/probe") if os.path.exists("checkpoints/probe") else "None")
except Exception as e:
    print("Kernels output error:", e)
