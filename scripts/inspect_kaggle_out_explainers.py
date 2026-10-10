import json
import glob
import os
from PIL import Image

for case_dir in sorted(glob.glob("kaggle_out/explainers/*")):
    print(f"\nDirectory: {case_dir}")
    fj = os.path.join(case_dir, "findings.json")
    if os.path.exists(fj):
        with open(fj) as f:
            print("Findings:", json.load(f))
    for png in sorted(glob.glob(f"{case_dir}/*.png")):
        im = Image.open(png)
        print(f"  {os.path.basename(png)}: size={im.size}, mode={im.mode}")
