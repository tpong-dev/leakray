# expect: LK009
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


def clean(frame):
    frame = frame.drop_duplicates()
    frame["income"].fillna(frame["income"].median(), inplace=True)
    frame["age"] = frame["age"].fillna(frame["age"].mean())
    return frame


survey = clean(pd.read_csv("survey.csv"))
X = survey[["income", "age", "household"]]
y = survey["spend"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=12)
print(LGBMRegressor().fit(X_tr, y_tr).score(X_te, y_te))
