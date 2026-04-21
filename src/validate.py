import pandas as pd
from pathlib import Path

DATA = Path("data/processed/creditcard_clean.csv")
OUT = Path("reports/validation.txt")

OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

lines = []
lines.append(f"rows={len(df)}")
lines.append(f"cols={len(df.columns)}")
lines.append(f"duplicates={df.duplicated().sum()}")
lines.append(f"missing_values={df.isnull().sum().sum()}")
lines.append(f"class_0={(df['Class'] == 0).sum()}")
lines.append(f"class_1={(df['Class'] == 1).sum()}")
lines.append(f"time_min={df['Time'].min()}")
lines.append(f"time_max={df['Time'].max()}")
lines.append(f"amount_min={df['Amount'].min()}")
lines.append(f"amount_max={df['Amount'].max()}")

OUT.write_text("\n".join(lines) + "\n")
print("Wrote:", OUT)