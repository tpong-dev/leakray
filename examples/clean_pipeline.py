"""Leak-free version of leaky_pipeline.py."""

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data.csv", parse_dates=["date"]).sort_values("date")
df = df.fillna(0)  # constant fill: nothing is learned from the data

# Split first, chronologically: train < validation < test.
n = len(df)
train = df.iloc[: int(n * 0.6)]
valid = df.iloc[int(n * 0.6) : int(n * 0.8)]
test = df.iloc[int(n * 0.8) :]

features = [c for c in df.columns if c not in ("target", "date")]
X_train, y_train = train[features], train["target"]
X_valid, y_valid = valid[features], valid["target"]
X_test, y_test = test[features], test["target"]

# Scaling and resampling are fitted on the training data only.
pipe = Pipeline(
    [
        ("scale", StandardScaler()),
        ("smote", SMOTE()),
        ("model", XGBClassifier()),
    ]
)
pipe.fit(X_train, y_train, model__eval_set=[(X_valid, y_valid)])
print(pipe.score(X_test, y_test))
