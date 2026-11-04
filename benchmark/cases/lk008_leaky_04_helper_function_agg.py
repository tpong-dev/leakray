# expect: LK008
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def add_brand_rate(frame):
    frame["brand_rate"] = frame.groupby("brand")["units_sold"].transform("mean")
    return frame


sales = add_brand_rate(pd.read_csv("retail.csv").drop_duplicates())
X = sales[["brand_rate", "price", "discount"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, sales["units_sold"])
print(RandomForestRegressor().fit(X_tr, y_tr).score(X_te, y_te))
