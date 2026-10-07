with open('scripts/training/train_kaggle_notebook.py', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'def assemble_labels' in line:
            print(f"Line {i+1}: {line.strip()}")
