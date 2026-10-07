from pathlib import Path
from ucimlrepo import fetch_ucirepo

OUT_PATH = Path("data/heart.csv")

heart = fetch_ucirepo(id=45)
df = heart.data.features.copy()
df["target"] = (heart.data.targets["num"] > 0).astype(int)

before = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Rows: {before} -> {len(df)} after dropping missing values")

for col in ["ca", "thal"]:
    df[col] = df[col].astype(int)

OUT_PATH.parent.mkdir(exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Saved {OUT_PATH}")