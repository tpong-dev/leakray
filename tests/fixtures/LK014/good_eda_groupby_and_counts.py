import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("orders.csv").drop_duplicates()
print(df.groupby("customer")["amount"].mean().sort_values().tail())
print(df["city"].value_counts(normalize=True).head())
summary = df.groupby("city")["amount"].agg(["mean", "count"])
summary.to_csv("city_summary.csv")
X_tr, X_te, y_tr, y_te = train_test_split(df[["amount"]], df["returned"])
