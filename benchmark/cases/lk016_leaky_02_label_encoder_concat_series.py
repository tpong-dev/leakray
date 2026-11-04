# expect: LK016
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv")
le = LabelEncoder()
le.fit(pd.concat([train["neighbourhood"], test["neighbourhood"]]))
train["neighbourhood"] = le.transform(train["neighbourhood"])
test["neighbourhood"] = le.transform(test["neighbourhood"])
