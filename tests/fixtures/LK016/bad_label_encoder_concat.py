import pandas as pd
from sklearn.preprocessing import LabelEncoder
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
le = LabelEncoder()
le.fit(pd.concat([train["city"], test["city"]]))
