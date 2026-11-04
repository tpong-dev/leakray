# expect: LK010
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

grid = pd.read_csv("grid_load.csv").drop_duplicates()
y = grid["load"]
grid["load_ewm"] = grid["load"].ewm(span=24).mean()
X = grid[["load_ewm", "temperature"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False)
model = GradientBoostingRegressor()
model.fit(X_tr, y_tr)
print(model.score(X_te, y_te))
