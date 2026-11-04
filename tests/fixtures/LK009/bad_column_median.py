import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("d.csv").drop_duplicates()
df["age"] = df["age"].fillna(df["age"].median())
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
