# expect: LK010
import pandas as pd
from catboost import CatBoostRegressor

grid = pd.read_csv("grid_load.csv").drop_duplicates().reset_index(drop=True)
series = grid.assign(
    trend=lambda d: d["load_mw"].ewm(span=12).mean(),
    delta=lambda d: d["load_mw"].diff(),
).dropna()

X = series[["trend", "delta", "temp_c"]]
y = series["load_mw"]
split_at = int(len(series) * 0.75)
model = CatBoostRegressor(verbose=0).fit(X[:split_at], y[:split_at])
print(model.score(X[split_at:], y[split_at:]))
