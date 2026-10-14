from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

kernel_slug = "nswitzer/creaky-apex-cleanlab-blend"
competition_slug = "rsna-knee-abnormality-detection"
message = "E22: Apex Grandmaster N4 Synergy (384px CoAtNet Anchor + Full 3-Fold Track N4 Suite)"

print("Submitting kernel version 4 of creaky-apex-cleanlab-blend...")
try:
    res = api.competition_submit_code(
        file_name="submission.csv",
        message=message,
        competition=competition_slug,
        kernel=kernel_slug,
        kernel_version=4
    )
    print("SUCCESS! Submission response:", res)
except Exception as e:
    print(f"Submission failed: {e}")
    # Inspect exception details if available
    if hasattr(e, 'response') and hasattr(e.response, 'text'):
        print("Response text:", e.response.text)
