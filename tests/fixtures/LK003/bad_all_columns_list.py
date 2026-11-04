import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("houses.csv")
features = df.columns.tolist()
X = df[features]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
RandomForestRegressor().fit(X_train, y_train)
