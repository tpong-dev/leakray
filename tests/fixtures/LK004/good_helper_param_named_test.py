import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def scale(train, test):
    sc = MinMaxScaler().fit(train)
    return sc.transform(train), sc.transform(test)


df = pd.read_csv("houses.csv")
a, b, ya, yb = train_test_split(df[["sqft"]], df["price"])
a, b = scale(a, b)
print(Ridge().fit(a, ya).score(b, yb))
