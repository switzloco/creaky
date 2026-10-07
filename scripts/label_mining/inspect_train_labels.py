import pandas as pd

df = pd.read_csv('data/processed/train_labels.csv')
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(2))
