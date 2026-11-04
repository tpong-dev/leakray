# expect: LK003
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

TARGET = "is_fraud"
df = pd.read_csv("transactions.csv").drop_duplicates()
X = df.drop(columns=["merchant_name", "card_type"])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
clf = LGBMClassifier().fit(X_train, y_train)
print(clf.score(X_test, y_test))
