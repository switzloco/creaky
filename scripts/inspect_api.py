import sys
from kaggle.api import kaggle_api_extended

print("Module attributes:")
for name in dir(kaggle_api_extended):
    if "Kernel" in name or "Output" in name or "Request" in name:
        print(" ", name)
