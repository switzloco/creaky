import json

nb_path = 'notebooks/ensemble/creaky-0-94-ensemble.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_str = 'CHECKPOINT_FILTER: List[str] = ["creaky-e11-reportaux-fold0"]'
new_str = 'CHECKPOINT_FILTER: List[str] = ["creaky-e11-reportaux-fold0", "creaky-e11-reportaux-fold1", "creaky-e11-reportaux-fold2"]'

found = False
for cell in nb['cells']:
    if isinstance(cell.get('source'), list):
        for idx, line in enumerate(cell['source']):
            if old_str in line:
                cell['source'][idx] = line.replace(old_str, new_str)
                found = True
                print("Replaced in list line!")
    elif isinstance(cell.get('source'), str):
        if old_str in cell['source']:
            cell['source'] = cell['source'].replace(old_str, new_str)
            found = True
            print("Replaced in str source!")

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated creaky-0-94-ensemble.ipynb with 3-fold filter!")
else:
    print("Warning: old_str not found!")
