"""Verify generated sample volume."""
import os
import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/sample_processed/series_metadata.csv")
print("=== SERIES METADATA ===")
print(df.iloc[0].to_dict())

out_path = os.path.join("data/processed/sample_processed", df["Output_File"][0])
arr = np.load(out_path)
print("\n=== NUMPY VOLUME ===")
print(f"Shape: {arr.shape} (N_slices x H x W)")
print(f"Dtype: {arr.dtype}")
print(f"Value Range: [{arr.min()}, {arr.max()}]")
print(f"Mean Intensity: {arr.mean():.2f}")
