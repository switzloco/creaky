from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

res = api.competition_submit_code(
    file_name="submission.csv",
    message="Speed-Demon V19 Hyper-Sprint (Kernel v21): In-Process Zero-Subprocess + True 2-GPU Sharded 4-Model Engine (cuda0=R384+N4, cuda1=R224+C384) + 8-Proc study_lut",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/creaky-apex-cleanlab-blend",
    kernel_version=21,
)
print("Submitted:", res)

subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in subs[:6]:
    print(f"ref={s.ref} | date={s.date} | status={s.status} | score={s.public_score} | desc={s.description[:75]}")
