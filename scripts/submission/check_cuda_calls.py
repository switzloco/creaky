import json

with open('notebooks/ensemble/creaky-0-94-ensemble.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Checking hardcoded .cuda() calls in ensemble notebook...")
hardcoded = []
for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    for line in src.splitlines():
        if '.cuda()' in line and 'if' not in line:
            hardcoded.append((idx, line.strip()))

print(f"Found {len(hardcoded)} potential hardcoded .cuda() calls:")
for c_idx, line in hardcoded[:10]:
    print(f"  Cell {c_idx}: {line}")
