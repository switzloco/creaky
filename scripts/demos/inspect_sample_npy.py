import numpy as np

path = 'data/processed/sample_processed/1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191_1.2.826.0.1.3680043.8.498.11580656442259111255675562605155903947.npy'
arr = np.load(path)
print("Shape:", arr.shape)
print("Dtype:", arr.dtype)
print("Min:", arr.min(), "Max:", arr.max(), "Mean:", arr.mean())
