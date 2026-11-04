# expect: LK001
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

houses = pd.read_csv("houses.csv")
houses = houses.drop_duplicates()
features = houses[["sqft", "rooms", "age"]]
target = houses["price"]
scaler = MinMaxScaler()
scaler.fit(features)
features = scaler.transform(features)
Xtr, Xte, ytr, yte = train_test_split(features, target, random_state=0)
reg = RandomForestRegressor(n_estimators=200)
reg.fit(Xtr, ytr)
print(reg.score(Xte, yte))
