import os
import glob
from PIL import Image
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

image_files = glob.glob("kaggle_out*/**/*.png", recursive=True) + glob.glob("scratch/*.png")
print(f"Total PNG images found: {len(image_files)}")
for img_path in sorted(image_files)[:30]:
    try:
        im = Image.open(img_path)
        print(f"  {img_path} | Size: {im.size} | Mode: {im.mode}")
    except Exception as e:
        print(f"  {img_path} | Error: {e}")
