# expect: clean
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

shops = pd.read_csv("shop_week.csv").drop_duplicates()
print(shops.groupby("store")["price"].mean().sort_values().tail())
print(shops["store"].value_counts().head())
X = shops[["price", "discount"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, shops["units"])
print(GradientBoostingRegressor().fit(X_tr, y_tr).score(X_te, y_te))
