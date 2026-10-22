"""Submit Speed-Demon V20 Safe-Turbo (Kernel Version 22) to RSNA Knee MRI competition."""
from kaggle.api.kaggle_api_extended import KaggleApi


def main() -> None:
    api = KaggleApi()
    api.authenticate()
    comp = "rsna-knee-abnormality-detection"
    desc = (
        "Speed-Demon V20 Safe-Turbo (Kernel v22): Pre-CUDA Overlapped DICOM Prefetch + "
        "Parallel 2-GPU Init + 50/50 Study Sharding + Fused 48-Slice N4 (0.947 Math Parity)"
    )
    res = api.competition_submit_code(
        file_name="submission.csv",
        competition=comp,
        kernel="nswitzer/creaky-apex-cleanlab-blend",
        kernel_version=22,
        submission_description=desc,
    )
    print("Submitted:", res)
    subs = api.competition_submissions(comp)
    for s in subs[:6]:
        print(
            f"ref={getattr(s, 'ref', '')} | date={getattr(s, 'date', '')} | "
            f"status={getattr(s, 'status', '')} | score={getattr(s, 'public_score', '')} | "
            f"desc={getattr(s, 'description', '')[:75]}"
        )


if __name__ == "__main__":
    main()
