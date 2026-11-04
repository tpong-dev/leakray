import pandas as pd
from sklearn.model_selection import train_test_split

orders = pd.read_csv("orders.csv")
orders = orders.sort_values("order_date")
X = orders[["order_date", "basket_size"]]
y = orders["revenue"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
