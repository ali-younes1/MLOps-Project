import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

RAW = Path("data/raw/creditcard.csv")
OUT = Path("data/processed/creditcard_clean.csv")

df = pd.read_csv(RAW)

# remove duplicates
df = df.drop_duplicates()

# remove missing values
df = df.dropna()

# scale Amount and Time
scaler = RobustScaler()
df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Wrote:", OUT)