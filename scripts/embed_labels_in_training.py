"""Embed gzipped train_labels.csv into scripts/training/train_kaggle_notebook.py."""
import gzip
import base64
from pathlib import Path

csv_path = Path("data/processed/train_labels.csv")
if not csv_path.exists():
    raise FileNotFoundError("data/processed/train_labels.csv not found")

data = csv_path.read_bytes()
comp = gzip.compress(data)
b64_str = base64.b64encode(comp).decode("ascii")
print(f"Compressed size: {len(comp):,} bytes | Base64 len: {len(b64_str):,} chars")

script_path = Path("scripts/training/train_kaggle_notebook.py")
content = script_path.read_text(encoding="utf-8")

# Marker to insert or replace embedded labels
start_marker = "# === EMBEDDED_JEV_LABELS_START ==="
end_marker = "# === EMBEDDED_JEV_LABELS_END ==="

payload_code = f'{start_marker}\nEMBEDDED_JEV_LABELS_B64 = "{b64_str}"\n{end_marker}'

if start_marker in content and end_marker in content:
    pre = content[:content.index(start_marker)]
    post = content[content.index(end_marker) + len(end_marker):]
    new_content = pre + payload_code + post
else:
    # Insert right before assemble_labels definition
    idx = content.find("def assemble_labels(")
    if idx != -1:
        new_content = content[:idx] + payload_code + "\n\n" + content[idx:]
    else:
        new_content = content + "\n\n" + payload_code

script_path.write_text(new_content, encoding="utf-8")
print(f"Successfully embedded Jev labels into {script_path}!")
