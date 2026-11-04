import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
TARGET = "price"
df = pd.read_csv("houses.csv")
features = ["sqft", "rooms", TARGET]
X = df[features]
y = df[TARGET]
RandomForestRegressor().fit(X, y)
