import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
model = XGBClassifier(n_estimators=1000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
