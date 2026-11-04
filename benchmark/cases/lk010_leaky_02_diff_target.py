# expect: LK010
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("demand.csv").drop_duplicates()
df["demand_change"] = df["demand"].diff()
df = df.dropna()
X = df[["demand_change", "price"]]
y = df["demand"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, shuffle=False, test_size=0.2)
print(Ridge().fit(X_tr, y_tr).score(X_te, y_te))
