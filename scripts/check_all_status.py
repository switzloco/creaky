from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

for k in ["nswitzer/submission-notebook", "nswitzer/training-book"]:
    status = api.kernels_status(k)
    print(f"{k} status:", getattr(status, "__dict__", status))
