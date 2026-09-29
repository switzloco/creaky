import os
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# list files in kernel output
req = api.kernels_output("nswitzer/training-book", path="checkpoints/probe_tb")
print("Files in training-book output:")
for root, dirs, files in os.walk("checkpoints/probe_tb"):
    for f in files:
        fp = os.path.join(root, f)
        print(f"  {fp} ({os.path.getsize(fp):,} bytes)")
