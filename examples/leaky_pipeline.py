"""Example pipeline with several data leakage bugs (for leakray demos)."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data.csv", parse_dates=["date"])

# Leak: imputing with statistics of the full dataset before splitting.
df = df.fillna(df.mean(numeric_only=True))

X = df.drop(columns=["target", "date"])
y = df["target"]

# Leak: scaler fitted on all rows, including future test rows.
X = StandardScaler().fit_transform(X)

# Leak: oversampling before the split puts synthetic copies in the test set.
X, y = SMOTE().fit_resample(X, y)

# Leak: random split on time-ordered data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBClassifier()
# Leak: early stopping on the test set.
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
print(model.score(X_test, y_test))
