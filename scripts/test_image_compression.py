import os
import io
import base64
from PIL import Image

candidates = [
    ("case1_raw", "kaggle_out_v9/explainers/case_1/ACL_raw.png"),
    ("case1_highlight", "kaggle_out_v9/explainers/case_1/ACL_highlight.png"),
    ("case1_normal", "kaggle_out_v9/explainers/case_1/normal_reference.png"),
    ("case2_raw", "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_raw.png"),
    ("case2_highlight", "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_highlight.png"),
    ("case3_raw", "kaggle_out_v9/explainers/case_3/ACL_raw.png"),
    ("case3_highlight", "kaggle_out_v9/explainers/case_3/ACL_highlight.png"),
    ("case4_raw", "kaggle_out_v9/explainers/case_4/Contusion_raw.png"),
    ("case4_highlight", "kaggle_out_v9/explainers/case_4/Contusion_highlight.png"),
    ("case5_raw", "kaggle_out_v9/explainers/case_5/PF_OA_raw.png"),
    ("case5_highlight", "kaggle_out_v9/explainers/case_5/PF_OA_highlight.png"),
    ("case6_mm_highlight", "kaggle_out/explainers/case_10002/Medial_Meniscus_highlight.png"),
    ("case7_eff_highlight", "kaggle_out/explainers/case_10004/Effusion_highlight.png"),
    ("case8_oa_highlight", "kaggle_out/explainers/case_10005/Medial_OA_highlight.png"),
    ("norm_sagittal", "scratch/healthy_Sagittal.png"),
    ("norm_axial", "scratch/healthy_Axial.png"),
    ("norm_coronal", "scratch/healthy_Coronal.png")
]

compressed_payload = {}
for name, p in candidates:
    im = Image.open(p)
    buf = io.BytesIO()
    # Save as JPEG with quality=88 (clean for medical gray-levels and colored overlays)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.save(buf, format="JPEG", quality=88, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    compressed_payload[name] = b64
    print(f"{name:20s}: {len(buf.getvalue())/1024:.1f} KB (b64 length: {len(b64)})")

total_kb = sum(len(b) for b in compressed_payload.values()) / 1024
print(f"\nTotal compressed base64 payload size: {total_kb:.1f} KB")
