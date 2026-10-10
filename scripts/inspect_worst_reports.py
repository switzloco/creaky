import pandas as pd
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

train_df = pd.read_csv("data/raw/train.csv")
series_df = pd.read_csv("data/raw/train_series.csv")

uids = [
    "1.2.826.0.1.3680043.8.498.13267780356245120052411517053322874891",
    "1.2.826.0.1.3680043.8.498.12505035424093604269515328931488770819",
    "1.2.826.0.1.3680043.8.498.56512564464301544960446419143135447363",
    "1.2.826.0.1.3680043.8.498.75187434248356774277526985329346125190",
    "1.2.826.0.1.3680043.8.498.16060119389060497136231217718921482192",
    "1.2.826.0.1.3680043.8.498.37392127307867238524251864751713143683",
    "1.2.826.0.1.3680043.8.498.48348615349418615469875801439348424274",
    "1.2.826.0.1.3680043.8.498.18392509497170616983977319528036573378",
    "1.2.826.0.1.3680043.8.498.13335129881737731410081002627729200903",
    "1.2.826.0.1.3680043.8.498.32321830776739689645700555055955725945"
]

for uid in uids:
    print(f"\n=================== Study: {uid} ===================")
    t_row = train_df[train_df["StudyInstanceUID"] == uid]
    if len(t_row) > 0:
        row = t_row.iloc[0]
        print("Gold labels:", {k: row[k] for k in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'] if pd.notna(row[k])})
        print("Report:\n", str(row['Report']).strip())
    s_rows = series_df[series_df["StudyInstanceUID"] == uid]
    print("Series count:", len(s_rows))
    for _, s in s_rows.iterrows():
        print(f"  Plane: {s['Anatomical_Plane']:8s} | Fluid: {s['Fluid_Sensitive']} | FatSup: {s['Fat_Suppression']} | SeriesUID: {s['SeriesInstanceUID']}")
