"""Check gzipped size of train_labels.csv."""
import gzip
import base64
from pathlib import Path

raw_path = Path("data/processed/train_labels.csv")
if raw_path.exists():
    data = raw_path.read_bytes()
    compressed = gzip.compress(data)
    b64 = base64.b64encode(compressed).decode("ascii")
    print(f"Raw CSV size:        {len(data):,} bytes ({len(data)/1024:.1f} KB)")
    print(f"Gzip size:           {len(compressed):,} bytes ({len(compressed)/1024:.1f} KB)")
    print(f"Base64 string size:  {len(b64):,} chars")
