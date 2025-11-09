# preview_csv.py
import pandas as pd
from pathlib import Path

# auto-find csvs under repo
csvs = list(Path('.').rglob('*.csv'))
print("Found CSV files:")
for c in csvs:
    print(" -", c)

# change this to the exact path you want after seeing the list above
p = csvs[0] if csvs else None
if not p:
    print("No CSV files found under the current folder.")
else:
    print("\nPreviewing:", p)
    df = pd.read_csv(p, low_memory=False)
    print("Columns:", list(df.columns))
    print(df.head(10).to_string(index=False))
