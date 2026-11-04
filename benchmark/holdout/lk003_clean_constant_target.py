# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

TARGET = "is_fraud"
tx = pd.read_csv("transactions.csv").drop_duplicates()
X, y = tx.drop(columns=TARGET), tx[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=11)
booster = LGBMClassifier(learning_rate=0.05).fit(X_tr, y_tr)
print(booster.score(X_te, y_te))
