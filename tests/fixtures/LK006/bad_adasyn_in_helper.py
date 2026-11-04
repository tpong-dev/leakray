import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split


def balance(features, labels):
    return ADASYN().fit_resample(features, labels)


df = pd.read_csv("events.csv")
Xb, yb = balance(df.drop(columns="event"), df["event"])
X_tr, X_te, y_tr, y_te = train_test_split(Xb, yb)
