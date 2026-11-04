import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)
model = LGBMRegressor()
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
print(model.score(X_test, y_test))
