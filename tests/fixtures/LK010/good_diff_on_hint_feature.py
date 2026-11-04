import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
df = pd.read_csv("demand.csv", parse_dates=["date"]).sort_values("date")
df["price_change"] = df["price"].diff()
df["revenue_pct"] = df["revenue"].pct_change()
y = df["units"]
X_tr, X_te, y_tr, y_te = train_test_split(df[["price_change", "revenue_pct"]].fillna(0), y, shuffle=False)
