import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
df = pd.read_csv("genes.csv")
X = df.drop(columns=["label"])
y = df["label"]
rfe = RFE(LogisticRegression(), n_features_to_select=10)
X_small = rfe.fit_transform(X, y)
print(cross_val_score(LogisticRegression(), X_small, y))
