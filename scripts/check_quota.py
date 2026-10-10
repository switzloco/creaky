from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

res = api.competition_get_submission_limits('rsna-knee-abnormality-detection')
print("Submission limits:", res)
