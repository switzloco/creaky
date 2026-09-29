import json
import re

with open("checkpoints/e02_trained/training-book.log", "r", encoding="utf-8") as f:
    text = f.read()

events = json.loads(text)
for e in events:
    d = e.get("data", "")
    if "Gold   AUC:" in d or "Epoch" in d and "Summary" in d or "Train Loss" in d or "Saved NEW BEST" in d:
        print(d.strip())
