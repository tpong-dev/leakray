import pandas as pd
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
all_data = pd.concat([train, test])
sc = StandardScaler()
sc.fit(all_data[["age", "income"]])
train_s = sc.transform(train[["age", "income"]])
