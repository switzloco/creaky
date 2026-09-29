import subprocess

msg = "feat: Phase 2 prep - 5-fold multilabel stratification & volume-consistent augmentation; Phase 3 label audit & domain notes"
subprocess.run(["git", "add", "DOMAIN_NOTES.md", "scripts/"], cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
result = subprocess.run(["git", "commit", "-m", msg], capture_output=True, text=True, cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
print("COMMIT STDOUT:", result.stdout)
print("COMMIT STDERR:", result.stderr)

if result.returncode == 0:
    push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True, cwd=r"c:\Users\nswitzer\Antigrav Proj\creaky")
    print("PUSH STDOUT:", push.stdout)
    print("PUSH STDERR:", push.stderr)
