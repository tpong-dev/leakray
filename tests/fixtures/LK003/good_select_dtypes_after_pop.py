import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("insurance.csv")
y = df.pop("charges")
X = df.select_dtypes(include="number")
RandomForestRegressor().fit(X, y)
