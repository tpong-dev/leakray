# expect: LK015
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load(path):
    frame = pd.read_csv(path)
    return frame.drop(columns="label"), frame["label"]


X, y = load("scraped_pages.csv")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, test_size=0.25)
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
