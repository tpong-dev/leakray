# expect: LK008
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("rentals.csv").drop_duplicates()
df["city_te"] = df.groupby("city")["rent"].transform("mean")
X = df[["city_te", "sqft", "bedrooms"]]
y = df["rent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(GradientBoostingRegressor().fit(X_train, y_train).score(X_test, y_test))
