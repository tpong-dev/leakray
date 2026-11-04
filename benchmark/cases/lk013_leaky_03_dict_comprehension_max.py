# expect: LK013
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("yield.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="tons"), df["tons"])
candidates = {"ridge": Ridge(), "gbr": GradientBoostingRegressor(), "et": ExtraTreesRegressor()}
results = {name: est.fit(X_train, y_train).score(X_test, y_test) for name, est in candidates.items()}
best = max(results, key=results.get)
print(best, results[best])
