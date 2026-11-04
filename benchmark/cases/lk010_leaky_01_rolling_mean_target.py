# expect: LK010
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("store_sales.csv").drop_duplicates()
y = df["sales"]
df["sales_ma7"] = y.rolling(7).mean()
df = df.dropna()
X = df[["sales_ma7", "promo"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, df["sales"], shuffle=False)
print(RandomForestRegressor().fit(X_tr, y_tr).score(X_te, y_te))
