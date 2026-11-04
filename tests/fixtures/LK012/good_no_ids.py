import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv").drop_duplicates()
X = df.drop(columns=["species"])
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
