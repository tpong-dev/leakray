import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
df = pd.read_csv("visits.csv", parse_dates=["date"]).sort_values("date")
df = df.assign(visits_ma=lambda d: d["visitors"].rolling(7).mean())
y = df["visitors"]
X_tr, X_te, y_tr, y_te = train_test_split(df[["visits_ma"]].fillna(0), y, shuffle=False)
