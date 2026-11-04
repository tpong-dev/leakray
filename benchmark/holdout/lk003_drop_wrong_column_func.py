# expect: LK003
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def split_xy(frame, target="survived"):
    X = frame.drop(columns=["name", "ticket"])
    y = frame[target]
    return X, y


passengers = pd.read_csv("titanic_clean.csv").drop_duplicates()
feats, lbl = split_xy(passengers)
x_tr, x_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.3, random_state=3)
clf = LogisticRegression(max_iter=500).fit(x_tr, y_tr)
print(clf.score(x_te, y_te))
