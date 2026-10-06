"""Audit laterality mentions across all radiology reports in train.csv."""
import re
import pandas as pd
import unicodedata

def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

# Regex patterns for left/right across the dataset languages:
# ES: izquierda/derecha, izq/der
# NL/DE: links/rechts, li/re
# FR: gauche/droite, g/d
# TR: sol/sag
# EL: αριστερο/δεξι (aristero/dexi)
# BG/RU: лев/прав, ляв/десн

LEFT_PATTERNS = [
    r"\b(left|izquierd[ao]s?|izq|links|linke[rn]?|gauche|sol|αριστερ\w*|ляв\w*|лев\w*)\b"
]

RIGHT_PATTERNS = [
    r"\b(right|derech[ao]s?|der|rechts|rechte[rn]?|droit[es]?|sag|δεξ\w*|десн\w*|прав\w*)\b"
]

re_left = re.compile("|".join(LEFT_PATTERNS), re.IGNORECASE)
re_right = re.compile("|".join(RIGHT_PATTERNS), re.IGNORECASE)

df = pd.read_csv("data/raw/train.csv")
print(f"Total studies in train.csv: {len(df)}")
df = df.dropna(subset=["Report"]).copy()
print(f"Studies with Report: {len(df)}")

left_count = 0
right_count = 0
both_count = 0
neither_count = 0

for idx, row in df.iterrows():
    text = strip_accents(str(row["Report"]).lower())
    has_l = bool(re_left.search(text))
    has_r = bool(re_right.search(text))
    
    if has_l and has_r:
        both_count += 1
    elif has_l:
        left_count += 1
    elif has_r:
        right_count += 1
    else:
        neither_count += 1

print("\n--- Report Text Laterality Audit ---")
print(f"Right knee only:  {right_count:>5} ({right_count/len(df)*100:.1f}%)")
print(f"Left knee only:   {left_count:>5} ({left_count/len(df)*100:.1f}%)")
print(f"Both mentioned:   {both_count:>5} ({both_count/len(df)*100:.1f}%)")
print(f"Neither/Unknown:  {neither_count:>5} ({neither_count/len(df)*100:.1f}%)")
total_determined = left_count + right_count
if total_determined > 0:
    print(f"\nOf determined single-knee studies ({total_determined}):")
    print(f"  Right: {right_count/total_determined*100:.1f}%")
    print(f"  Left:  {left_count/total_determined*100:.1f}%")
