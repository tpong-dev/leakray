# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

df = pd.read_csv("energy.csv", parse_dates=["timestamp"]).drop_duplicates()
df = df.sort_values("timestamp")
df["hour"] = df["timestamp"].dt.hour
X = df[["hour", "temperature"]]
y = df["consumption"]
scores = cross_val_score(RandomForestRegressor(), X, y, cv=TimeSeriesSplit(n_splits=5))
print(scores.mean())
