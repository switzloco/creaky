import json
import glob

for f in sorted(glob.glob("kaggle_out_v9/explainers/case_*/findings.json")):
    print(f"=== {f} ===")
    with open(f, "r") as fp:
        data = json.load(fp)
        for k, v in data.items():
            if k != "output_files":
                print(f"  {k}: {v}")
