# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("store_sales.csv").drop_duplicates()
y = df["sales"]
df["sales_ma7"] = y.shift(1).rolling(7).mean()
df = df.dropna()
X_tr, X_te, y_tr, y_te = train_test_split(df[["sales_ma7", "promo"]], df["sales"], shuffle=False)
print(RandomForestRegressor().fit(X_tr, y_tr).score(X_te, y_te))
