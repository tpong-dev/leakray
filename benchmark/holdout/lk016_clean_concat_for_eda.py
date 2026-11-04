# expect: clean
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv").drop_duplicates()
num_cols = ["lot_area", "living_area", "garage_area"]

overview = pd.concat([train.assign(split="train"), test.assign(split="test")])
print(overview.groupby("split")[num_cols].describe())

scaler = StandardScaler().fit(train[num_cols])
model = LGBMRegressor().fit(scaler.transform(train[num_cols]), train["sale_price"])
preds = model.predict(scaler.transform(test[num_cols]))
