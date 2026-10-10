import os
import glob
from PIL import Image

for case_dir in sorted(glob.glob("kaggle_out_v9/explainers/*")):
    if os.path.isdir(case_dir):
        print(f"\nDirectory: {case_dir}")
        for f in sorted(os.listdir(case_dir)):
            fp = os.path.join(case_dir, f)
            if f.endswith(".png"):
                im = Image.open(fp)
                print(f"  {f:25s} | {im.size} | {im.mode}")
            elif f.endswith(".json"):
                print(f"  {f:25s} | (json metadata)")
            elif f.endswith(".html"):
                print(f"  {f:25s} | (html view)")
