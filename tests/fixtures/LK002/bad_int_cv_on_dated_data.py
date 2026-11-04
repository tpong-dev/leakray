import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

load = pd.read_csv("load.csv")
load["created_at"] = pd.to_datetime(load["created_at"])
X = load[["temp", "humidity"]]
y = load["kwh"]
print(cross_val_score(LinearRegression(), X, y, cv=5))
