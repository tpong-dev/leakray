import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
df = pd.read_csv("genes.csv").drop_duplicates()
X = df.drop(columns=["label"])
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
sel = SelectKBest(k=20).fit(X_train, y_train)
X_train, X_test = sel.transform(X_train), sel.transform(X_test)
