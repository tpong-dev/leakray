# expect: LK014
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

clicks = pd.read_csv("ad_clicks.csv").drop_duplicates()
freq = clicks["device_model"].value_counts(normalize=True)
clicks["device_freq"] = clicks["device_model"].map(freq)
X = clicks[["device_freq", "hour_of_day", "position"]]
X_train, X_test, y_train, y_test = train_test_split(X, clicks["clicked"], stratify=clicks["clicked"])
print(XGBClassifier().fit(X_train, y_train).score(X_test, y_test))
