import pandas as pd
from sklearn.preprocessing import StandardScaler
train_a = pd.read_csv("train_a.csv")
train_b = pd.read_csv("train_b.csv")
train = pd.concat([train_a, train_b])
StandardScaler().fit(train[["age"]])
