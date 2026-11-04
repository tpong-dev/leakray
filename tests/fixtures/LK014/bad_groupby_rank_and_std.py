import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("orders.csv").drop_duplicates()
df["amt_std"] = df.groupby("customer")["amount"].transform("std")
df["amt_rank"] = df.groupby("customer")["amount"].rank(pct=True)
X_tr, X_te, y_tr, y_te = train_test_split(df[["amt_std", "amt_rank"]], df["returned"])
