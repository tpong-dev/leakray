import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

houses = pd.read_csv("houses.csv")
X = houses[["rooms", "area", "age"]]
y = houses["price"]
print(cross_val_score(Ridge(), X, y, cv=KFold(5, shuffle=True, random_state=0)).mean())
