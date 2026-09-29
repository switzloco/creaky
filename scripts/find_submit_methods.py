import inspect
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
for name in dir(api):
    if "submit" in name.lower() or "kernel" in name.lower():
        print(name)
