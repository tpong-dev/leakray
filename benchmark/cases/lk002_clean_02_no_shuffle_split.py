# expect: clean
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sales = pd.read_csv("sales.csv").drop_duplicates()
sales["ts"] = pd.to_datetime(sales["ts"])
sales = sales.sort_values("ts")
X = sales[["promo", "price"]]
y = sales["units"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(Ridge().fit(X_train, y_train).score(X_test, y_test))
