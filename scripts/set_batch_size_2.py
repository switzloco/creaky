import json

p = "notebooks/training-book-c384-gpu-f1/training-book-c384-gpu-f1.ipynb"
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell_source = "".join(nb["cells"][1]["source"])
cell_source = cell_source.replace('"batch_size": 1,', '"batch_size": 2,')
cell_source = cell_source.replace('"accum_steps": 4,', '"accum_steps": 2,')

# Validate syntax
compile(cell_source, "<gpu_nb>", "exec")

nb["cells"][1]["source"] = [line for line in cell_source.splitlines(keepends=True)]

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated GPU notebook to batch_size=2, accum_steps=2 and verified python syntax!")
