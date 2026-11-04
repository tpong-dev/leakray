# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge


def scale(train, test):
    sc = MinMaxScaler().fit(train)
    return sc.transform(train), sc.transform(test)


houses = pd.read_csv("houses.csv").drop_duplicates()
a, b, ya, yb = train_test_split(houses[["sqft", "rooms"]], houses["price"], random_state=3)
a, b = scale(a, b)
print(Ridge().fit(a, ya).score(b, yb))
