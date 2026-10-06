from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

status = api.kernels_status("nswitzer/training-book")
print("Status dict:", status)
print("Attributes:", dir(status))
for attr in ["status", "error", "message", "has_failure_message", "failure_message"]:
    if hasattr(status, attr):
        print(f"  {attr}: {getattr(status, attr)}")

