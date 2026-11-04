# expect: LK002
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

sales = pd.read_csv("sales.csv").drop_duplicates()
sales["ts"] = pd.to_datetime(sales["ts"])
sales["dow"] = sales["ts"].dt.dayofweek
X = sales[["dow", "promo", "price"]]
y = sales["units"]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
print(cross_val_score(Ridge(), X, y, cv=cv).mean())
