# expect: LK006
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def balance(features, labels):
    return ADASYN(n_neighbors=3).fit_resample(features, labels)


df = pd.read_csv("rare_events.csv").drop_duplicates()
Xb, yb = balance(df.drop(columns="event"), df["event"])
X_tr, X_te, y_tr, y_te = train_test_split(Xb, yb, stratify=yb)
print(SVC().fit(X_tr, y_tr).score(X_te, y_te))
