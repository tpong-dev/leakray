# expect: LK004
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv")
cats = ["color", "size", "brand"]
enc_train = OneHotEncoder(handle_unknown="ignore").fit(train[cats])
X_train = enc_train.transform(train[cats])
X_test = OneHotEncoder(handle_unknown="ignore").fit_transform(test[cats])
print(X_train.shape, X_test.shape)
