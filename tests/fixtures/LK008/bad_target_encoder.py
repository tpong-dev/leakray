import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
df = pd.read_csv("listings.csv").drop_duplicates()
X = df.drop(columns=["price"])
y = df["price"]
enc = TargetEncoder(cols=["city"])
X = enc.fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
