# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv("credit.csv").drop_duplicates()
X, y = df.drop(columns="bad"), df["bad"]
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, stratify=y_tmp)
model = XGBClassifier(n_estimators=2000, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("final test acc", model.score(X_test, y_test))
