# expect: LK003
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv").drop_duplicates()
y = df["charges"]
X = df.select_dtypes(include="number")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9)
gbr = GradientBoostingRegressor().fit(X_train, y_train)
print(gbr.score(X_test, y_test))
