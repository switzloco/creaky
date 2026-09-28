"""Pull metadata from existing Kaggle kernels."""
import subprocess

for slug in ["nswitzer/training-book", "nswitzer/submission-notebook"]:
    print(f"Pulling metadata for {slug}...")
    folder = "notebooks/" + slug.split("/")[1]
    res = subprocess.run(["uv", "run", "kaggle", "kernels", "pull", slug, "-p", folder, "-m"], capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)
