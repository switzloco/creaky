"""Check existing Kaggle kernels."""
import subprocess

res = subprocess.run(["uv", "run", "kaggle", "kernels", "list", "--mine"], capture_output=True, text=True)
print("STDOUT:\n", res.stdout)
