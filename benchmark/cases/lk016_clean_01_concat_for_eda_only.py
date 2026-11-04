# expect: clean
import pandas as pd
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv")
both = pd.concat([train.assign(split="train"), test.assign(split="test")])
print(both.groupby("split")[["f1", "f2"]].describe())
scaler = StandardScaler().fit(train[["f1", "f2"]])
X_test = scaler.transform(test[["f1", "f2"]])
