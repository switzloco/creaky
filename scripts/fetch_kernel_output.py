import os
import sys
import argparse
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

def fetch_output(kernel: str, target_dir: str):
    api = KaggleApi()
    api.authenticate()

    os.makedirs(target_dir, exist_ok=True)
    print(f"Connecting to Kaggle API to download outputs for {kernel}...")
    owner_slug, kernel_slug, version = api.parse_kernel_string(kernel)

    with api.build_kaggle_client() as kaggle:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = owner_slug
        request.kernel_slug = kernel_slug
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)

    files = response.files or []
    print(f"Found {len(files)} output files:")
    for item in files:
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
        print(f"Saved {kernel_slug}.log ({len(response.log)} chars).")
        # Print tail of log
        lines = response.log.strip().splitlines()
        print("\n--- Recent Log Lines ---")
        for line in lines[-20:]:
            print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="nswitzer/training-book")
    parser.add_argument("--dir", default="checkpoints/e04_training")
    args = parser.parse_args()
    fetch_output(args.kernel, args.dir)
