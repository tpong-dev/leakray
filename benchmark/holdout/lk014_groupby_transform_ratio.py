# expect: LK014
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

orders = pd.read_csv("orders.csv").drop_duplicates()
orders["amt_vs_cat"] = orders["amount"] / orders.groupby("category")["amount"].transform("mean")
orders["cat_std"] = orders.groupby("category")["amount"].transform("std")

X = orders[["amt_vs_cat", "cat_std", "items"]]
y = orders["returned"]
data_tr, data_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=6)
print(CatBoostClassifier(verbose=0).fit(data_tr, y_tr).score(data_te, y_te))
