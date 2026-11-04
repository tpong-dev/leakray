import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("orders.csv").drop_duplicates()
df["amount"] = df["amount"] / df["amount"].max()
X_tr, X_te, y_tr, y_te = train_test_split(df[["amount"]], df["returned"])
