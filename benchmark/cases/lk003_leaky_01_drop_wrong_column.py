# expect: LK003
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("cars.csv").drop_duplicates()
X = df.drop(columns=["model_name"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
