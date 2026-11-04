import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("reviews.csv").drop_duplicates()
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
