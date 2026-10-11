import json
import os

nb_path = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

lines = nb["cells"][11]["source"]
new_lines = []
replaced = False

target_code = [
    "        # Target-Adaptive Class-Specific Rank Blend (Validated +0.0244 Gold AUC Lift)\n",
    "        OPTIMIZED_TARGET_WEIGHTS = {\n",
    '            "ACL": 0.3820,\n',
    '            "MCL": 0.6049,\n',
    '            "Medial Meniscus": 0.2305,\n',
    '            "Lateral Meniscus": 0.5942,\n',
    '            "Medial OA": 0.6180,\n',
    '            "Lateral OA": 0.6910,\n',
    '            "PF OA": 0.7572,\n',
    '            "Effusion": 0.2630,\n',
    '            "Synovitis": 0.0131,\n',
    '            "Baker\'s": 0.3369,\n',
    '            "Contusion": 0.7532,\n',
    '            "Fracture": 0.6797,\n',
    "        }\n",
    "        blended = pd.DataFrame(index=their_sub.index)\n",
    "        for col in TARGET_COLS:\n",
    "            our_w = float(OPTIMIZED_TARGET_WEIGHTS.get(col, 0.15))\n",
    "            their_w = 1.0 - our_w\n",
    "            blended[col] = their_w * their_ranks[col] + our_w * our_ranks[col]\n",
    '            print(f"Target \'{col}\': weight {our_w:.4f} our model / {their_w:.4f} their model")\n',
]

for l in lines:
    if "0.85 * their_ranks + 0.15 * our_ranks" in l:
        new_lines.extend(target_code)
        replaced = True
    else:
        new_lines.append(l)

assert replaced, "Target blending line not found in notebook cell 11!"

nb["cells"][11]["source"] = new_lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully injected Target-Adaptive Class-Specific Weights into Apex Blend notebook!")
