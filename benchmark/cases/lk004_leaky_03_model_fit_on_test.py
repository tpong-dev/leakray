# expect: LK004
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("bikes.csv").drop_duplicates()
X = df[["temp", "wind", "season"]]
y = df["rentals"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = DecisionTreeRegressor(max_depth=6)
reg.fit(X_test, y_test)
print("R2 on holdout:", reg.score(X_test, y_test))
