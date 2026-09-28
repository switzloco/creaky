import json
from pathlib import Path

nb_path = Path("kaggle samples/maverick/rsna-knee-restructured-version-3.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell["cell_type"]
    source = "".join(cell["source"]).strip()
    if ctype == "markdown":
        print(f"\n--- [Cell {i} Markdown] ---")
        print(source[:300])
    elif "class " in source or "def " in source or "CONFIG" in source or "class " in source:
        print(f"\n--- [Cell {i} Code Structure] ---")
        for line in source.splitlines():
            line_s = line.strip()
            if any(line_s.startswith(x) for x in ["class ", "def ", "CONFIG", "MODEL", "class", "import timm", "import torch"]):
                print("  ", line_s)
            elif "timm.create_model" in line or "nn.Linear" in line or "nn.Conv" in line or "loss" in line.lower() and "=" in line:
                print("  -->", line_s[:100])

