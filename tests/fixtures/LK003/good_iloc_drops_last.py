import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("houses.csv")
X = df.iloc[:, :-1].values
y = df["price"].values
RandomForestRegressor().fit(X, y)
