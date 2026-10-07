from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

subs = api.competition_submissions("rsna-knee-abnormality-detection")
if subs:
    latest = subs[0]
    print(f"Latest submission (Ref: {latest.ref}):")
    for k, v in getattr(latest, "__dict__", {}).items():
        print(f"  {k}: {v}")
