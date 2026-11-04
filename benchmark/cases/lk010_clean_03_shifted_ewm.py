# expect: clean
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

grid = pd.read_csv("grid_load.csv").drop_duplicates()
grid["load_ewm_prev"] = grid["load"].shift().ewm(span=24).mean()
grid = grid.dropna()
y = grid["load"]
X_tr, X_te, y_tr, y_te = train_test_split(grid[["load_ewm_prev", "temperature"]], y, shuffle=False)
print(GradientBoostingRegressor().fit(X_tr, y_tr).score(X_te, y_te))
