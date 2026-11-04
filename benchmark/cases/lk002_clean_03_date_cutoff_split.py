# expect: clean
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

orders = pd.read_csv("orders.csv", parse_dates=["order_date"]).drop_duplicates()
orders["month"] = orders["order_date"].dt.month
cutoff = pd.Timestamp("2024-01-01")
train = orders[orders["order_date"] < cutoff]
test = orders[orders["order_date"] >= cutoff]
feats = ["month", "basket_size"]
model = HistGradientBoostingRegressor().fit(train[feats], train["revenue"])
print(model.score(test[feats], test["revenue"]))
