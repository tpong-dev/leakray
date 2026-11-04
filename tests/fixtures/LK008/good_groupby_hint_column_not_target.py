import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("shop.csv").drop_duplicates()
df["store_avg_price"] = df.groupby("store")["price"].transform("mean")
y = df["returned"]
X_tr, X_te, y_tr, y_te = train_test_split(df[["store_avg_price", "qty"]], y)
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
