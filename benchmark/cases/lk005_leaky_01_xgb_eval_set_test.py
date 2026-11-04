# expect: LK005
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("credit.csv").drop_duplicates()
X, y = df.drop(columns="bad"), df["bad"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
model = XGBClassifier(n_estimators=2000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(model.score(X_test, y_test))
