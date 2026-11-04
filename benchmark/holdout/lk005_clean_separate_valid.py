# expect: clean
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

df_all = pd.read_csv("insurance.csv").drop_duplicates()
feats, lbl = df_all.drop(columns="claim"), df_all["claim"]
X_rest, X_test, y_rest, y_test = train_test_split(feats, lbl, test_size=0.2, random_state=0)
X_fit, X_val, y_fit, y_val = train_test_split(X_rest, y_rest, test_size=0.2, random_state=1)

cb = CatBoostClassifier(iterations=3000, early_stopping_rounds=100, verbose=0)
cb.fit(X_fit, y_fit, eval_set=(X_val, y_val))
print(cb.score(X_test, y_test))
