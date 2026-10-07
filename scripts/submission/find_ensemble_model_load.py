import json

with open('notebooks/ensemble/creaky-0-94-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = cell.get('source', '')
    if isinstance(src, list):
        src = "".join(src)
    if "CHECKPOINT_FILTER" in src:
        print(f"Found in Cell {i}!")
        for l in src.splitlines():
            if "CHECKPOINT_FILTER" in l:
                print("  Line:", repr(l))
