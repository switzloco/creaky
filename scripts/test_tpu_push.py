import json
from pathlib import Path

meta_path = Path("scratch/test_tpu_meta.json")
meta = {
    "id": "nswitzer/test-tpu-quota-check",
    "title": "Test TPU Quota Check",
    "code_file": "scratch/test_tpu.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": True
}
meta_path.write_text(json.dumps(meta, indent=2))
print("Wrote test TPU metadata")
