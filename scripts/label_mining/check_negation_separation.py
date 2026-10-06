"""Task T4 Step 2: Sanity check comparing E11 char TF-IDF+SVD fingerprints vs E5-small embeddings.

Tests whether E5-small separates negated pairs (e.g. 'tear' vs 'no tear') better than TF-IDF.
Under TF-IDF character n-grams, 'rotura de LCA' and 'sin rotura de LCA' share 90%+ identical n-grams,
yielding artificially high cosine similarity (>0.85).
E5-small semantic representations should distinctly lower cosine similarity for opposite diagnoses.
"""
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sentence_transformers import SentenceTransformer

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

# 5 representative pairs across languages: positive finding vs negated/normal finding
test_pairs = [
    (
        "ES - LCA Rotura",
        "rotura completa del ligamento cruzado anterior con derrame articular asociado.",
        "ligamento cruzado anterior integro sin signos de rotura ni derrame articular."
    ),
    (
        "ES - Menisco",
        "rotura del cuerno posterior del menisco medial alcanzando la superficie articular inferior.",
        "menisco medial de morfologia y senal normal, sin evidencia de rotura meniscal."
    ),
    (
        "EN - ACL Tear",
        "complete tear of the anterior cruciate ligament with joint effusion and bone marrow edema.",
        "intact anterior cruciate ligament without evidence of tear, effusion, or osseous injury."
    ),
    (
        "NL - Meniscus",
        "scheur van het achterhoorn van de mediale meniscus met gewrichtsuitstorting.",
        "geen scheur van de mediale meniscus, normale morfologie zonder gewrichtsuitstorting."
    ),
    (
        "FR - LCA",
        "rupture complete du ligament croise anterieur associee a un epanchement intra-articulaire.",
        "integrite du ligament croise anterieur sans signe de rupture ni epanchement articulaire."
    )
]

print("--- Computing E11 TF-IDF + SVD Fingerprints (dim=64) on train corpus ---")
df = pd.read_csv("data/raw/train.csv").dropna(subset=["Report"])
texts = [strip_accents(str(t)).lower() for t in df["Report"]]

tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=60000,
                        min_df=3, sublinear_tf=True, dtype=np.float32)
X = tfidf.fit_transform(texts)
svd = TruncatedSVD(n_components=64, random_state=42)
svd.fit(X)

print("\n--- Loading multilingual-e5-small for comparison ---")
model = SentenceTransformer("intfloat/multilingual-e5-small")

# Read generated PCA from dataset
embed_df = pd.read_csv("data/processed/report_embed_e5_64.csv")
print(f"Loaded {len(embed_df)} precomputed E5-64 vectors.")

print("\n" + "="*80)
print(f"{'Test Pair':<20} | {'E11 TF-IDF Cosine':<18} | {'E5-small Cosine':<16} | {'Delta':<10}")
print("="*80)

for label, p_pos, p_neg in test_pairs:
    # 1. TF-IDF vector
    v_pos_tfidf = svd.transform(tfidf.transform([strip_accents(p_pos).lower()]))[0]
    v_neg_tfidf = svd.transform(tfidf.transform([strip_accents(p_neg).lower()]))[0]
    v_pos_tfidf /= (np.linalg.norm(v_pos_tfidf) + 1e-8)
    v_neg_tfidf /= (np.linalg.norm(v_neg_tfidf) + 1e-8)
    cos_tfidf = float(np.dot(v_pos_tfidf, v_neg_tfidf))
    
    # 2. E5 raw vector
    e_pos = model.encode([f"passage: {strip_accents(p_pos).lower()}"], normalize_embeddings=True)[0]
    e_neg = model.encode([f"passage: {strip_accents(p_neg).lower()}"], normalize_embeddings=True)[0]
    cos_e5 = float(np.dot(e_pos, e_neg))
    
    delta = cos_tfidf - cos_e5
    print(f"{label:<20} | {cos_tfidf:>17.4f} | {cos_e5:>15.4f} | {delta:>+9.4f}")

print("="*80)
print("Note: Positive delta means E5 separates negated concepts better than character TF-IDF.")
