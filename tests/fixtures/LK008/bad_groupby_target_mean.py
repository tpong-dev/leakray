import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("listings.csv").drop_duplicates()
y = df["price"]
df["city_price"] = df.groupby("city")["price"].transform("mean")
X = df.drop(columns=["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
