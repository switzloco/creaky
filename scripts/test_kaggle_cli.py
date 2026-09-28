"""Test Kaggle CLI capabilities."""
import subprocess

def run():
    print("Testing Kaggle CLI...")
    res = subprocess.run(["uv", "run", "kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)

if __name__ == "__main__":
    run()
