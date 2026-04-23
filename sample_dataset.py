import pandas as pd
import os

df = pd.read_csv("data/processed/creditcard_clean.csv")
df_sample = df.sample(n=5000, random_state=42)

os.makedirs("data/sample", exist_ok=True)

df_sample.to_csv("data/sample/creditcard_sample.csv", index=False)