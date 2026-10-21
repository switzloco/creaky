from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

res = api.competition_submit_code(
    file_name="submission.csv",
    message="Speed-Demon V18 Ultra-Sprint (Kernel v20): Fused Single-Read 4-Model Ensemble (Raptor-384 + Raptor-224 + C384 CoAtNet F0 + N4 ConvNeXt F0, Zero 2nd-Pass IO, ~3.5m)",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/creaky-apex-cleanlab-blend",
    kernel_version=20,
)
print("Submitted:", res)

subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in subs[:6]:
    print(f"ref={s.ref} | date={s.date} | status={s.status} | score={s.public_score} | desc={s.description[:75]}")
