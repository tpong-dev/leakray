import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
TARGET = "price"
df = pd.read_csv("houses.csv")
features = df.columns.drop([TARGET, "id"])
X = df.loc[:, features]
y = df[TARGET]
RandomForestRegressor().fit(X, y)
