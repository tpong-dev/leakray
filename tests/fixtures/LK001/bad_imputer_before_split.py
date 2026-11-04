import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
df = pd.read_csv("houses.csv")
X = df.drop(columns=["price"])
y = df["price"]
imp = SimpleImputer(strategy="median")
X = imp.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
