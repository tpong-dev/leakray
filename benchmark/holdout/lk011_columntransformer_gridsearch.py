# expect: LK011
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

bank = pd.read_csv("bank_marketing.csv").drop_duplicates()
lbl = bank["subscribed"]
feats = bank.drop(columns="subscribed")

prep = ColumnTransformer([
    ("num", RobustScaler(), ["balance", "duration", "campaign"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["job", "marital"]),
])
X_prep = prep.fit_transform(feats)

search = GridSearchCV(
    LGBMClassifier(),
    param_grid={"num_leaves": [15, 31, 63], "learning_rate": [0.03, 0.1]},
    cv=5,
    scoring="roc_auc",
)
search.fit(X_prep, lbl)
print(search.best_score_)
