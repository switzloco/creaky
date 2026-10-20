import inspect
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print("Signature:", inspect.signature(api.competition_submit_code))

res = api.competition_submit_code(
    file_name="submission.csv",
    message="Speed-Demon V14 (Kernel v15): Zero-Second-Pass DICOM IO (/tmp/s2_cache 0.2ms) + All-GPU 2.5D Slab Builder + K48 Stride-2 Dual-T4 (~7-9 min target)",
    competition="rsna-knee-abnormality-detection",
    kernel="nswitzer/creaky-apex-cleanlab-blend",
    kernel_version=15,
)
print("Submitted:", res)

subs = api.competition_submissions("rsna-knee-abnormality-detection")
for s in subs[:6]:
    print(f"ref={s.ref} | date={s.date} | status={s.status} | score={s.public_score} | desc={s.description[:75]}")
