import json
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import IPython.display

captured_outputs = []

def custom_display(*args, **kwargs):
    for a in args:
        data_str = getattr(a, "data", None)
        if data_str is not None:
            captured_outputs.append({
                "data": {
                    "text/html": [data_str]
                },
                "execution_count": 1,
                "metadata": {},
                "output_type": "display_data"
            })
        else:
            captured_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [str(a) + "\n"]
            })

IPython.display.display = custom_display

for path in [
    "doctor_mri_error_audit.ipynb",
    "notebooks/doctor_mri_error_audit.ipynb"
]:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Execute Cell 1
    captured_outputs.clear()
    c1_code = "".join(nb["cells"][1]["source"])
    exec(c1_code, globals())
    nb["cells"][1]["outputs"] = list(captured_outputs)
    nb["cells"][1]["execution_count"] = 1

    # Execute Cell 3
    captured_outputs.clear()
    c3_code = "".join(nb["cells"][3]["source"])
    exec(c3_code, globals())
    nb["cells"][3]["outputs"] = list(captured_outputs)
    nb["cells"][3]["execution_count"] = 2

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"Pre-rendered outputs saved to {path} ({len(nb['cells'][3]['outputs'])} outputs, file size: {round(len(json.dumps(nb))/1024, 1)} KB)")

