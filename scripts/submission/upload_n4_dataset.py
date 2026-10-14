import os
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def upload():
    # Fix Windows temp path issue for Kaggle CLI
    temp_upload = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / ".kaggle" / "uploads"
    temp_upload.mkdir(parents=True, exist_ok=True)
    
    # Also ensure relative path is simple
    upload_dir = Path("data/upload/creaky-n4-cleanlab-models").resolve()
    
    api = KaggleApi()
    api.authenticate()
    
    print(f"Uploading dataset from {upload_dir} to Kaggle...")
    try:
        res = api.dataset_create_new(
            folder=str(upload_dir),
            public=False,
            quiet=False
        )
        print("Dataset creation response:", res)
    except Exception as e:
        print(f"Error during dataset create: {e}")
        # If already exists or error, try version update
        try:
            res = api.dataset_create_version(
                folder=str(upload_dir),
                version_notes="Track N4 Cleanlab Folds 0, 1, 2",
                quiet=False
            )
            print("Dataset version update response:", res)
        except Exception as v_err:
            print(f"Version update error: {v_err}")

if __name__ == "__main__":
    upload()
