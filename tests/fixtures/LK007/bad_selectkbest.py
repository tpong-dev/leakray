import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
df = pd.read_csv("genes.csv").drop_duplicates()
X = df.drop(columns=["label"])
y = df["label"]
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sel, y)
