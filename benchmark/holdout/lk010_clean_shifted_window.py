# expect: clean
import pandas as pd
from lightgbm import LGBMRegressor

sales = pd.read_csv("store_units.csv").drop_duplicates().reset_index(drop=True)
past = sales["units"].shift(1)
sales["units_avg7"] = past.rolling(7).mean()
sales["units_lag1"] = past
sales["promo_avg3"] = sales["promo"].rolling(3).mean()
sales = sales.dropna()

feats = sales[["units_avg7", "units_lag1", "promo_avg3"]]
lbl = sales["units"]
cut = int(len(sales) * 0.8)
reg = LGBMRegressor().fit(feats.iloc[:cut], lbl.iloc[:cut])
print(reg.score(feats.iloc[cut:], lbl.iloc[cut:]))
