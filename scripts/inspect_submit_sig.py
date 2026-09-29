import inspect
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

sig = inspect.signature(api.competition_submit)
print("competition_submit signature:", sig)
