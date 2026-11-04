# expect: LK002
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

orders = pd.read_csv("orders.csv", parse_dates=["order_date"])
orders = orders.drop_duplicates()
orders = orders.assign(month=orders["order_date"].dt.month, year=orders["order_date"].dt.year)
X = orders[["month", "year", "basket_size"]]
y = orders["revenue"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=True, test_size=0.15)
model = XGBRegressor(n_estimators=300).fit(X_tr, y_tr)
print(model.score(X_te, y_te))
