import io
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

subs = api.competition_submissions("rsna-knee-abnormality-detection")
print(f"Total submissions: {len(subs)}")
for s in subs[:5]:
    print(f"Ref: {s.ref}, Date: {s.date}, Status: {s.status}, PublicScore: {s.public_score}, Description: {s.description}")
