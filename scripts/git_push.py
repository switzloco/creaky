import subprocess

msg = "feat: 0.885 Public LB (+0.003) - preprocessing parity fix verified on Kaggle, launch E02 training"
subprocess.run(["git", "add", "EXPERIMENTS.md", "PLAN.md", "scripts/"], cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
result = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True, cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
print("COMMIT STDOUT:", result.stdout)
print("COMMIT STDERR:", result.stderr)

if result.returncode == 0:
    push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
    print("PUSH STDOUT:", push.stdout)
    print("PUSH STDERR:", push.stderr)
