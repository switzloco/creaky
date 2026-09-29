import inspect
from kaggle.api.kaggle_api_extended import KaggleApi

print("competition_submit_code:", inspect.signature(KaggleApi.competition_submit_code))
