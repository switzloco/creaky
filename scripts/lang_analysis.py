"""Analyze the 58 gold-labeled reports to check language and finding patterns."""
import pandas as pd
import re

train = pd.read_csv("data/raw/train.csv")
labeled = train.dropna(subset=["ACL"])

print(f"Total labeled: {len(labeled)}")

target_cols = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

# Let's inspect some language clues in gold set
def detect_lang(text):
    text = str(text).lower()
    if "antecedentes" in text or "hallazgos" in text or "rotura" in text or "derrame" in text or "rodilla" in text:
        return "Spanish"
    elif "constatations" in text or "alignement" in text or "dgnratifs" in text or "genou" in text:
        return "French"
    elif "inlichtingen" in text or "bevindingen" in text or "scheur" in text or "gewogen" in text:
        return "Dutch"
    elif "findings" in text or "impression" in text or "tear" in text or "intact" in text:
        return "English"
    elif "befund" in text or "beurteilung" in text:
        return "German"
    else:
        return "Other/Mixed"

labeled["lang"] = labeled["Report"].apply(detect_lang)
print("Gold set language distribution:")
print(labeled["lang"].value_counts())

train["lang"] = train["Report"].apply(detect_lang)
print("\nAll 4407 studies language distribution (rough):")
print(train["lang"].value_counts())
