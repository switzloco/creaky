from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

res = api.competition_submit_code(
    file_name="submission.csv",
    message="Speed-Demon V17 (Kernel v19): Exact V13 0.949 Parity + Unmounted OrthoFoundation + Parallel 8-Model Loader + cuDNN Benchmark + Bounded 12-Study GPU Slab Builder (~14m)",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/creaky-apex-cleanlab-blend",
    kernel_version=19,
)
print("Submitted:", res)

subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in subs[:6]:
    print(f"ref={s.ref} | date={s.date} | status={s.status} | score={s.public_score} | desc={s.description[:75]}")
