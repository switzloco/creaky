import os
import glob
import base64
from PIL import Image
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Collect all available images
candidates = [
    # Case 1: ACL / Effusion discrepancy
    ("case1_raw", "kaggle_out_v9/explainers/case_1/ACL_raw.png"),
    ("case1_highlight", "kaggle_out_v9/explainers/case_1/ACL_highlight.png"),
    ("case1_normal", "kaggle_out_v9/explainers/case_1/normal_reference.png"),
    
    # Case 2: Lateral Meniscus / Medial Meniscus confounder
    ("case2_raw", "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_raw.png"),
    ("case2_highlight", "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_highlight.png"),
    
    # Case 3: ACL + Medial Meniscus combo
    ("case3_raw", "kaggle_out_v9/explainers/case_3/ACL_raw.png"),
    ("case3_highlight", "kaggle_out_v9/explainers/case_3/ACL_highlight.png"),
    
    # Case 4: Contusion & Effusion
    ("case4_raw", "kaggle_out_v9/explainers/case_4/Contusion_raw.png"),
    ("case4_highlight", "kaggle_out_v9/explainers/case_4/Contusion_highlight.png"),
    
    # Case 5: Patellofemoral Osteoarthritis (PF OA) Missed
    ("case5_raw", "kaggle_out_v9/explainers/case_5/PF_OA_raw.png"),
    ("case5_highlight", "kaggle_out_v9/explainers/case_5/PF_OA_highlight.png"),
    
    # Additional cases from kaggle_out
    ("case6_mm_highlight", "kaggle_out/explainers/case_10002/Medial_Meniscus_highlight.png"),
    ("case7_eff_highlight", "kaggle_out/explainers/case_10004/Effusion_highlight.png"),
    ("case8_oa_highlight", "kaggle_out/explainers/case_10005/Medial_OA_highlight.png"),
    
    # Healthy benchmarks from scratch
    ("norm_sagittal", "scratch/healthy_Sagittal.png"),
    ("norm_axial", "scratch/healthy_Axial.png"),
    ("norm_coronal", "scratch/healthy_Coronal.png")
]

print("Verifying image files:")
valid_images = {}
for name, p in candidates:
    if os.path.exists(p):
        size_kb = os.path.getsize(p) / 1024
        im = Image.open(p)
        print(f"  [OK] {name:20s}: {p} | {im.size} | {size_kb:.1f} KB")
        with open(p, "rb") as f:
            valid_images[name] = base64.b64encode(f.read()).decode("ascii")
    else:
        print(f"  [MISSING] {name:20s}: {p}")

print(f"\nTotal valid images: {len(valid_images)}")
total_b64_kb = sum(len(b) for b in valid_images.values()) / 1024
print(f"Total base64 payload size: {total_b64_kb:.1f} KB")

