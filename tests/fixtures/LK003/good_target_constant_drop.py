import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
TARGET = "price"
df = pd.read_csv("houses.csv")
X = df.drop(TARGET, axis=1)
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y)
RandomForestRegressor().fit(X_train, y_train)
