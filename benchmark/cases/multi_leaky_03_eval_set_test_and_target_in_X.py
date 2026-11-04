# expect: LK003, LK005
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("taxi_fares.csv").drop_duplicates()
X = df.drop(columns=["vendor_name"])
y = df["fare"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LGBMRegressor(n_estimators=1000)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
