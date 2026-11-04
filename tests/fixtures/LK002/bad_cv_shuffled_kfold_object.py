import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("traffic.csv")
df["ts"] = pd.to_datetime(df["ts"])
df["hour"] = df["ts"].dt.hour
X = df[["hour", "lanes"]]
y = df["volume"]
splitter = KFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(Ridge(), X, y, cv=splitter).mean())
