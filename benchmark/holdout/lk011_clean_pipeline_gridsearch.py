# expect: clean
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

bank = pd.read_csv("bank_marketing.csv").drop_duplicates()
lbl = bank["subscribed"]
feats = bank.drop(columns="subscribed")

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", RobustScaler(), ["balance", "duration", "campaign"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["job", "marital"]),
    ])),
    ("gbm", LGBMClassifier()),
])
search = GridSearchCV(pipe, {"gbm__num_leaves": [15, 31, 63]}, cv=5, scoring="roc_auc")
search.fit(feats, lbl)
print(search.best_score_)
