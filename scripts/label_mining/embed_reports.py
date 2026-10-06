"""Compute multilingual sentence embeddings for radiology reports using multilingual-e5-small.

Applies:
1. Passage prefix ('passage: ') as required by E5 models
2. Normalization
3. PCA dimensionality reduction from 384 -> 64 to maintain parameter parity with E11
4. L2 normalization of final vectors
"""
import os
import unicodedata
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

def main():
    train_csv = "data/raw/train.csv"
    out_csv = "data/processed/report_embed_e5_64.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    print(f"Loading {train_csv}...")
    df = pd.read_csv(train_csv)
    print(f"Total rows: {len(df)}")
    
    df = df.dropna(subset=["Report"]).copy()
    print(f"Rows with non-null Report: {len(df)}")
    
    # E5 models expect 'passage: ' prefix for document text
    raw_texts = [f"passage: {strip_accents(str(t)).lower().strip()}" for t in df["Report"]]
    study_uids = df["StudyInstanceUID"].astype(str).tolist()
    
    print("Loading multilingual-e5-small model...")
    model = SentenceTransformer("intfloat/multilingual-e5-small")
    
    print(f"Encoding {len(raw_texts)} reports on CPU (batch_size=32)...")
    embeddings = model.encode(raw_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    print(f"Raw embeddings shape: {embeddings.shape}")
    
    print("Applying PCA (384 -> 64 dims)...")
    pca = PCA(n_components=64, random_state=42)
    reduced_embeddings = pca.fit_transform(embeddings).astype(np.float32)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"Explained variance ratio (64 dims): {explained_var:.4f}")
    
    # L2 normalize reduced embeddings
    norms = np.linalg.norm(reduced_embeddings, axis=1, keepdims=True) + 1e-8
    reduced_embeddings /= norms
    
    # Build output dataframe
    col_names = [f"e{i}" for i in range(64)]
    out_df = pd.DataFrame(reduced_embeddings, columns=col_names)
    out_df.insert(0, "StudyInstanceUID", study_uids)
    
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {len(out_df)} embeddings to {out_csv}")

if __name__ == "__main__":
    main()
