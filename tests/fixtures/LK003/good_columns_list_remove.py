import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
TARGET = "price"
df = pd.read_csv("houses.csv")
features = list(df.columns)
features.remove(TARGET)
RandomForestRegressor().fit(df[features], df[TARGET])
