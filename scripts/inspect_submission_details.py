from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in subs:
    if s.ref == 56665264:
        print("Submission details:")
        for k, v in getattr(s, "__dict__", {}).items():
            print(f"  {k}: {v}")
