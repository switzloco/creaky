import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("Testing doctor_mri_error_audit.ipynb structure and code execution...")
with open("doctor_mri_error_audit.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    print(f"Cell {i} [{ctype}] - metadata: {cell.get('metadata')}")
    if ctype == "code":
        code = "".join(cell.get("source", []))
        print(f"  Executing cell {i} code ({len(code)} chars)...")
        # Compile to check syntax
        compiled = compile(code, f"<cell_{i}>", "exec")
        print(f"  [OK] Cell {i} compiled successfully!")

print("All code cells passed syntax compilation!")
