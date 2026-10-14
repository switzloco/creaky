import os
import json
from pathlib import Path

def stage():
    src_dir = Path("checkpoints/n4_cleanlab_trained")
    dst_dir = Path("data/upload/creaky-n4-cleanlab-models")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hardlink or copy .pt files
    for f in [0, 1, 2]:
        fn = f"best_model_fold_{f}.pt"
        src = src_dir / fn
        dst = dst_dir / fn
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
            print(f"Hardlinked {fn} to upload staging.")
        except Exception as e:
            import shutil
            shutil.copy2(src, dst)
            print(f"Copied {fn} to upload staging.")

    # 2. Dataset metadata
    meta = {
        "title": "Creaky Track N4 Cleanlab Models (Folds 0, 1, 2)",
        "id": "nswitzer/creaky-n4-cleanlab-models",
        "licenses": [{"name": "CC0-1.0"}]
    }
    meta_path = dst_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Written dataset-metadata.json to {meta_path}")

if __name__ == "__main__":
    stage()
