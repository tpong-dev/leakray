import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prep(data):
    return StandardScaler().fit_transform(data)


df = pd.read_csv("d.csv")
X_train, X_test = train_test_split(df)
X_test = prep(X_test)
