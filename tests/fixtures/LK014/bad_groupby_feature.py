import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("orders.csv").drop_duplicates()
y = df["returned"]
df["cust_avg_amount"] = df.groupby("customer")["amount"].transform("mean")
X = df.drop(columns=["returned"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
