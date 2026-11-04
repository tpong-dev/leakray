# expect: LK003
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

cols = ["sqft", "rooms", "age", "sale_price"]
houses = pd.read_csv("homes.csv").drop_duplicates()
X = houses[cols]
y = houses["sale_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
forest = RandomForestRegressor(n_estimators=200).fit(X_train, y_train)
print(forest.score(X_test, y_test))
