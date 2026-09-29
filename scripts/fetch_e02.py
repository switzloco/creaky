import os
import sys
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

api = KaggleApi()
api.authenticate()

kernel = "nswitzer/training-book"
target_dir = "checkpoints/e02_trained"
os.makedirs(target_dir, exist_ok=True)

print(f"Connecting to Kaggle API to safely download outputs for {kernel}...")
owner_slug, kernel_slug, version = api.parse_kernel_string(kernel)

with api.build_kaggle_client() as kaggle:
    request = ApiListKernelSessionOutputRequest()
    request.user_name = owner_slug
    request.kernel_slug = kernel_slug
    response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

print(f"Found {len(response.files or [])} output files:")
for item in response.files or []:
    outfile = os.path.join(target_dir, item.file_name)
    print(f"Downloading {item.file_name}...")
    resp = requests.get(item.url, stream=True)
    resp.raise_for_status()
    with open(outfile, "wb") as f:
        f.write(resp.content)
    sz = os.path.getsize(outfile) / (1024 * 1024)
    print(f"  --> Saved: {outfile} ({sz:.2f} MB)")

if response.log:
    log_file = os.path.join(target_dir, f"{kernel_slug}.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(response.log)
    print(f"Saved {kernel_slug}.log safely using UTF-8 ({len(response.log)} chars).")

print("\nAll downloads completed successfully!")
