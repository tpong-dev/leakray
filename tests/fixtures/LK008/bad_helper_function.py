import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def encode(frame, col):
    frame[col + "_te"] = frame.groupby(col)["converted"].transform("mean")
    return frame


leads = encode(pd.read_csv("leads.csv").drop_duplicates(), "source")
X = leads[["source_te", "visits"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, leads["converted"], stratify=leads["converted"])
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
