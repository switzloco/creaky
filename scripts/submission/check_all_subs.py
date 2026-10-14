import io
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

def check():
    api = KaggleApi()
    api.authenticate()
    subs = api.competition_submissions("rsna-knee-abnormality-detection")
    print(f"Total submissions retrieved: {len(subs)}")
    for s in subs[:10]:
        print(f"Ref: {s.ref} | Date: {s.date} | Status: {s.status} | Public Score: {s.public_score} | Desc: {s.description}")

if __name__ == "__main__":
    check()
