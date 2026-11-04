# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv("concrete.csv").drop_duplicates()
X, y = df.drop(columns="strength"), df["strength"]
cv = KFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(RandomForestRegressor(), X, y, cv=cv, scoring="r2").mean())
