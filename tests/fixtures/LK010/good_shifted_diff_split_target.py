import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
df = pd.read_csv("energy.csv", parse_dates=["ts"]).sort_values("ts")
df["load_delta_prev"] = df["load_mw"].shift(1).diff()
df = df.dropna()
X_tr, X_te, y_tr, y_te = train_test_split(df[["load_delta_prev"]], df["load_mw"], shuffle=False)
