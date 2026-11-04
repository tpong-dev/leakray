# expect: LK013
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

df_all = pd.read_csv("employees.csv").drop_duplicates()
feats, lbl = df_all.drop(columns="attrition"), df_all["attrition"]
X_tr, X_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.25, random_state=0)

zoo = {
    "logreg": LogisticRegression(max_iter=1000),
    "rf": RandomForestClassifier(),
    "lgbm": LGBMClassifier(),
    "cat": CatBoostClassifier(verbose=0),
}
results = {}
for name, est in zoo.items():
    est.fit(X_tr, y_tr)
    results[name] = accuracy_score(y_te, est.predict(X_te))
winner = max(results, key=results.get)
print("winner:", winner, results[winner])
