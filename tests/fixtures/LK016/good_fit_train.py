import pandas as pd
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sc = StandardScaler().fit(train[["age", "income"]])
test_s = sc.transform(test[["age", "income"]])
