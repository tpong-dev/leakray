import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit

prices = pd.read_csv("prices.csv", parse_dates=["day"], index_col="day")
X = prices[["volume", "spread"]]
y = prices["close"]
search = GridSearchCV(RandomForestRegressor(), {"max_depth": [3, 5]}, cv=ShuffleSplit(5, test_size=0.2))
search.fit(X, y)
