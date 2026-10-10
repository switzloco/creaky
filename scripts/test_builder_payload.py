import os
import io
import json
import base64
from PIL import Image

def get_compressed_b64(path, quality=88):
    if not os.path.exists(path):
        return ""
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")

print("Compressing images for payload...")
img_sources = {
    "c1_raw": "kaggle_out_v9/explainers/case_1/ACL_raw.png",
    "c1_heat": "kaggle_out_v9/explainers/case_1/ACL_highlight.png",
    "c2_raw": "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_raw.png",
    "c2_heat": "kaggle_out_v9/explainers/case_2/Lateral_Meniscus_highlight.png",
    "c3_raw": "kaggle_out_v9/explainers/case_3/ACL_raw.png",
    "c3_heat": "kaggle_out_v9/explainers/case_3/ACL_highlight.png",
    "c4_raw": "kaggle_out_v9/explainers/case_4/Contusion_raw.png",
    "c4_heat": "kaggle_out_v9/explainers/case_4/Contusion_highlight.png",
    "c5_raw": "kaggle_out_v9/explainers/case_5/PF_OA_raw.png",
    "c5_heat": "kaggle_out_v9/explainers/case_5/PF_OA_highlight.png",
    "c6_heat": "kaggle_out/explainers/case_10002/Medial_Meniscus_highlight.png",
    "c7_heat": "kaggle_out/explainers/case_10004/Effusion_highlight.png",
    "c8_heat": "kaggle_out/explainers/case_10005/Medial_OA_highlight.png",
    "ref_sag": "scratch/healthy_Sagittal.png",
    "ref_ax": "scratch/healthy_Axial.png",
    "ref_cor": "scratch/healthy_Coronal.png"
}

payload = {}
for k, path in img_sources.items():
    payload[k] = get_compressed_b64(path)
    print(f"  {k}: {len(payload[k])} chars")

total_size_kb = sum(len(v) for v in payload.values()) / 1024
print(f"Total payload size: {total_size_kb:.1f} KB")
