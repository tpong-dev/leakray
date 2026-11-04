# expect: LK016
import pandas as pd
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv")
num_cols = ["f1", "f2", "f3"]
full = pd.concat([train[num_cols], test[num_cols]])
scaler = StandardScaler().fit(full)
X_train = scaler.transform(train[num_cols])
X_test = scaler.transform(test[num_cols])
