from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

res = api.competition_submit_code(
    file_name="submission.csv",
    message="Speed-Demon V15 (Kernel v17): 50/50 Dual-GPU FLOP Sharding + Fused 48-Slice 3-Plane Batch + 4-Proc Fork Pool + Exact 0.949 Math",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/creaky-apex-cleanlab-blend",
    kernel_version=17,
)
print("Submitted:", res)

subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in subs[:6]:
    print(f"ref={s.ref} | date={s.date} | status={s.status} | score={s.public_score} | desc={s.description[:75]}")
