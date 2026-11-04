# expect: clean
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv")
cats = ["color", "size", "brand"]
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(train[cats])
X_train = encoder.transform(train[cats])
X_test = encoder.transform(test[cats])
print(X_train.shape, X_test.shape)
