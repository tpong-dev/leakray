import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("orders.csv").drop_duplicates()
df["amt_per_item"] = df["amount"] / df["items"].clip(lower=1)
train, test = train_test_split(df)
stats = train.groupby("customer")["amount"].agg(["mean"]).reset_index()
train = train.merge(stats, on="customer", how="left")
test = test.merge(stats, on="customer", how="left")
train["amount_n"] = train["amount"] / train["amount"].max()
