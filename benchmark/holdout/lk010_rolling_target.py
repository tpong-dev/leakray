# expect: LK010
import pandas as pd
from lightgbm import LGBMRegressor

sales = pd.read_csv("store_units.csv").drop_duplicates().reset_index(drop=True)
sales["units_avg7"] = sales["units"].rolling(7).mean()
sales["units_max7"] = sales["units"].rolling(window=7, min_periods=1).max()
sales = sales.dropna()

feats = sales[["units_avg7", "units_max7", "promo"]]
lbl = sales["units"]
cut = int(len(sales) * 0.8)
reg = LGBMRegressor().fit(feats.iloc[:cut], lbl.iloc[:cut])
print(reg.score(feats.iloc[cut:], lbl.iloc[cut:]))
