# expect: LK016
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv").drop_duplicates()
num_cols = ["lot_area", "living_area", "garage_area"]

full = pd.concat([train[num_cols], test[num_cols]], axis=0)
scaler = StandardScaler().fit(full)
X_train = scaler.transform(train[num_cols])
X_test = scaler.transform(test[num_cols])

model = LGBMRegressor().fit(X_train, train["sale_price"])
pd.DataFrame({"pred": model.predict(X_test)}).to_csv("submission.csv", index=False)
