# expect: LK002
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate

prices = pd.read_csv("prices.csv", parse_dates=["date"], index_col="date").drop_duplicates()
prices["ret_lag1"] = prices["close"].pct_change().shift(1)
prices = prices.dropna()
X = prices[["ret_lag1", "volume"]]
y = prices["close"].pct_change().fillna(0)
res = cross_validate(GradientBoostingRegressor(), X, y, cv=KFold(5, shuffle=True, random_state=1))
print(res["test_score"])
