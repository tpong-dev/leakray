# expect: clean
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("demand.csv").drop_duplicates()
df["price_change"] = df["price"].diff()
df["temp_roll"] = df["temperature"].rolling(3).mean()
df = df.dropna()
y = df["demand"]
X_tr, X_te, y_tr, y_te = train_test_split(df[["price_change", "temp_roll"]], y, shuffle=False)
print(Ridge().fit(X_tr, y_tr).score(X_te, y_te))
