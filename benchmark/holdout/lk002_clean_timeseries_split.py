# expect: clean
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

demand = (
    pd.read_csv("demand.csv", parse_dates=["week_start"])
    .drop_duplicates()
    .sort_values("week_start")
    .reset_index(drop=True)
)
X = demand[["price", "promo", "holiday_flag"]]
y = demand["qty"]
scores = cross_val_score(GradientBoostingRegressor(), X, y, cv=TimeSeriesSplit(n_splits=5))
print(scores.mean())

cutoff = int(len(demand) * 0.8)
X_hist, X_future = X.iloc[:cutoff], X.iloc[cutoff:]
y_hist, y_future = y.iloc[:cutoff], y.iloc[cutoff:]
final = GradientBoostingRegressor().fit(X_hist, y_hist)
print(final.score(X_future, y_future))
