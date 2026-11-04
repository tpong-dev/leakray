# expect: LK002
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

sales = pd.read_csv("daily_sales.csv", parse_dates=["order_date"]).drop_duplicates()
sales["dow"] = sales["order_date"].dt.dayofweek
sales["month"] = sales["order_date"].dt.month

feats = sales[["dow", "month", "promo", "store_size"]]
lbl = sales["units"]
f_tr, f_te, l_tr, l_te = train_test_split(feats, lbl, test_size=0.2, random_state=7)
reg = LGBMRegressor().fit(f_tr, l_tr)
print(reg.score(f_te, l_te))
