# expect: LK014
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

shops = pd.read_csv("shop_week.csv").drop_duplicates()
shops["store_price_std"] = shops.groupby("store")["price"].transform("std")
shops["store_price_rank"] = shops.groupby("store")["price"].rank(pct=True)
X = shops[["store_price_std", "store_price_rank", "discount"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, shops["units"])
print(GradientBoostingRegressor().fit(X_tr, y_tr).score(X_te, y_te))
