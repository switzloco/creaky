import json

with open("notebooks/training-book/training-book.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

found = []
for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if "print_split_report" in src:
        found.append("print_split_report")
    if "val_preds_path" in src:
        found.append("val_preds_path")
    if "cache_v1" in src:
        found.append("cache_v1")

print("Found keywords in training-book.ipynb:", set(found))
