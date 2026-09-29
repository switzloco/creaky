import subprocess

msg = "feat: implement Phase 2 - zero-dependency 5-fold multilabel stratification & vectorized volume-consistent augmentations"
subprocess.run(["git", "add", "-A"], cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
result = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True, cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
print("COMMIT STDOUT:", result.stdout)
print("COMMIT STDERR:", result.stderr)

if result.returncode == 0:
    push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
    print("PUSH STDOUT:", push.stdout)
    print("PUSH STDERR:", push.stderr)
