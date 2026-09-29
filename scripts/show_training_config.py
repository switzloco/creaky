import json

with open("notebooks/training-book/training-book.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if "CONFIG = {" in src:
        print(src[:src.find("}")+1])
        break
