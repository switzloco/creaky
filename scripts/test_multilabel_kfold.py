import io, gzip, base64, re
import numpy as np
import pandas as pd

def iterative_multilabel_split(df: pd.DataFrame, target_cols: list, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Zero-dependency iterative multilabel stratification (Seffke & Tsoumakas)."""
    np.random.seed(seed)
    n_samples = len(df)
    folds = np.full(n_samples, -1, dtype=int)
    
    # Binarize targets: >= 0.5 is positive
    Y = (df[target_cols].to_numpy() >= 0.5).astype(int)
    
    # Desired positive count per fold for each target
    c_counts = Y.sum(axis=0)
    desired_fold_counts = np.zeros((n_splits, len(target_cols)))
    for c in range(len(target_cols)):
        desired_fold_counts[:, c] = c_counts[c] / n_splits

    # Desired total samples per fold
    desired_samples = n_samples / n_splits
    fold_samples = np.zeros(n_splits)
    current_fold_counts = np.zeros((n_splits, len(target_cols)))

    # Sort labels by rarity (least frequent first)
    label_order = np.argsort(c_counts)

    # Sort samples by number of positive labels (descending), then random
    label_density = Y.sum(axis=1)
    sample_indices = np.argsort(-label_density)
    
    # Shuffle indices within each density group for randomness
    shuffled_indices = []
    for density in np.unique(label_density)[::-1]:
        group = sample_indices[label_density[sample_indices] == density]
        np.random.shuffle(group)
        shuffled_indices.extend(group)
    sample_indices = np.array(shuffled_indices)

    for idx in sample_indices:
        y = Y[idx]
        pos_labels = np.where(y == 1)[0]
        
        if len(pos_labels) > 0:
            # Pick the rarest positive label this sample has
            rarest = min(pos_labels, key=lambda l: c_counts[l])
            # Best fold is the one with the highest unmet need for this label
            shortfalls = desired_fold_counts[:, rarest] - current_fold_counts[:, rarest]
            best_fold = int(np.argmax(shortfalls))
        else:
            # Sample has no positives; assign to fold with fewest total samples
            shortfalls = desired_samples - fold_samples
            best_fold = int(np.argmax(shortfalls))
            
        folds[idx] = best_fold
        fold_samples[best_fold] += 1
        current_fold_counts[best_fold] += y

    return folds

if __name__ == "__main__":
    # Load labels
    with open("scripts/training/train_kaggle_notebook.py", "r", encoding="utf-8") as f:
        content = f.read()

    m = re.search(r'EMBEDDED_JEV_LABELS_B64\s*=\s*"([^"]+)"', content)
    b64 = m.group(1)
    raw = gzip.decompress(base64.b64decode(b64))
    df = pd.read_csv(io.BytesIO(raw))

    TARGET_COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

    # Stratify Silver studies (4,349 studies) across 5 folds
    silver_df = df[df["label_source"] == "jev"].reset_index(drop=True)
    silver_df["fold"] = iterative_multilabel_split(silver_df, TARGET_COLS, n_splits=5, seed=42)

    print("--- Silver Folds Size ---")
    print(silver_df["fold"].value_counts().sort_index())

    print("\n--- Positive Counts per Fold across 12 Targets ---")
    fold_pos = []
    for f in range(5):
        f_df = silver_df[silver_df["fold"] == f]
        f_dict = {"fold": f, "total": len(f_df)}
        for t in TARGET_COLS:
            f_dict[t] = int((f_df[t] >= 0.5).sum())
        fold_pos.append(f_dict)

    res_df = pd.DataFrame(fold_pos)
    print(res_df.to_string(index=False))

