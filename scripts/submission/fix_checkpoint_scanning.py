import json
import os

nb_path = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

lines = nb["cells"][11]["source"]

start_idx = None
end_idx = None
for i, l in enumerate(lines):
    if "# 2. Dataset-style attachments: scan 2 levels" in l:
        start_idx = i
    if start_idx is not None and "except OSError:" in l:
        end_idx = i + 2  # include pass
        break

assert start_idx is not None and end_idx is not None, "Could not find dataset scanning block!"

new_scan_code = [
    "        # 2. Dataset-style attachments: recursively scan all dataset/model dirs (skip raw competition DICOMs)\n",
    "        try:\n",
    '            for entry in os.listdir("/kaggle/input"):\n',
    '                if entry in ["competitions", "rsna-knee-abnormality-detection"]:\n',
    "                    continue  # skip DICOM competition directories\n",
    '                entry_path = os.path.join("/kaggle/input", entry)\n',
    "                if os.path.isdir(entry_path):\n",
    "                    for dirpath, _, filenames in os.walk(entry_path):\n",
    "                        for f in filenames:\n",
    '                            if f.endswith(".pt") or f.endswith(".pth"):\n',
    "                                ckpt_path = os.path.join(dirpath, f)\n",
    "                                ckpt_files.append(ckpt_path)\n",
    '                                print(f"  --> Found model checkpoint at: {ckpt_path}")\n',
    "        except OSError as e:\n",
    '            print(f"Error scanning checkpoints: {e}")\n',
]

new_lines = lines[:start_idx] + new_scan_code + lines[end_idx:]
nb["cells"][11]["source"] = new_lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully upgraded checkpoint scanning to full recursive search in Apex Blend notebook!")
