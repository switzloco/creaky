import io
import os
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

api = KaggleApi()
api.authenticate()

# 1. Verify GPU quota
try:
    quota = api.quota_view()
    print("Quota view:", quota)
except Exception as e:
    print("Could not query quota_view:", e)

print("\nChecking kernel status for nswitzer/creaky-0-94-ensemble...")
status = api.kernels_status("nswitzer/creaky-0-94-ensemble")
print("Current kernel status:", status)

print("\n--- STAGED ENSEMBLE KERNEL ---")
print("Notebook: notebooks/ensemble/creaky-0-94-ensemble.ipynb")
print("Target: nswitzer/creaky-0-94-ensemble Version 5")
print("Models included: creaky-e18-cleanlab-fold0 (0.8597 Macro AUC record) + creaky-e11-reportaux-fold1 + creaky-e11-reportaux-fold2")

def push_and_submit():
    print("\nPushing notebook to Kaggle...")
    os.system('cmd.exe /c "kaggle kernels push -p notebooks/ensemble"')
    
    print("\nWaiting for kernel push to register...")
    time.sleep(10)
    
    status = api.kernels_status("nswitzer/creaky-0-94-ensemble")
    print("Updated kernel status:", status)

if __name__ == "__main__":
    if "--execute" in sys.argv:
        push_and_submit()
    else:
        print("\nTo execute push & submit when quota refreshes, run with: --execute")
