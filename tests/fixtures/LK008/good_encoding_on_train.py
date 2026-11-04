import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("listings.csv").drop_duplicates()
y = df["price"]
X = df.drop(columns=["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
means = y_train.groupby(X_train["city"]).mean()
X_train["city_price"] = X_train["city"].map(means)
X_test["city_price"] = X_test["city"].map(means)
