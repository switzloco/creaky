"""Kaggle Synchronization and Leaderboard Tracker for Antigravity."""

import os
import sys
import subprocess
import argparse
import pandas as pd


def check_competition_submissions(competition: str = "rsna-knee-abnormality-detection"):
    """Fetch recent submissions and leaderboard scores."""
    print("=" * 70)
    print(f"CHECKING LEADERBOARD STATUS FOR: {competition}")
    print("=" * 70)
    cmd = f"uv run kaggle competitions submissions {competition}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr and "warning" not in res.stderr.lower():
        print(res.stderr)


def download_kernel_outputs(kernel_slug: str, output_dir: str = "data/kaggle_outputs"):
    """Download output files from a completed Kaggle notebook."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(f"DOWNLOADING OUTPUTS FROM: {kernel_slug} -> {output_dir}")
    print("=" * 70)
    cmd = f"uv run kaggle kernels output {kernel_slug} -p {output_dir}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Kaggle competition submissions and outputs")
    parser.add_argument("--action", choices=["submissions", "download"], default="submissions")
    parser.add_argument("--kernel", help="Kernel slug (e.g. nswitzer/creaky-inference)")
    parser.add_argument("--out_dir", default="data/kaggle_outputs")
    args = parser.parse_args()

    if args.action == "submissions":
        check_competition_submissions()
    elif args.action == "download":
        if not args.kernel:
            print("Error: --kernel required for download action.")
        else:
            download_kernel_outputs(args.kernel, args.out_dir)
