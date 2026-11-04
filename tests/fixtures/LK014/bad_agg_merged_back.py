import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("orders.csv").drop_duplicates()
stats = df.groupby("customer")["amount"].agg(["mean", "std"]).reset_index()
df = df.merge(stats, on="customer", how="left")
X_tr, X_te, y_tr, y_te = train_test_split(df[["mean", "std"]], df["returned"])
