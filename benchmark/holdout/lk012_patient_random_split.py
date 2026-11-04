# expect: LK012
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

visits = pd.read_csv("ecg_visits.csv").drop_duplicates()
feats = visits.drop(columns=["patient_id", "arrhythmia"])
lbl = visits["arrhythmia"]
X_tr, X_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.2, random_state=0)
print(LGBMClassifier().fit(X_tr, y_tr).score(X_te, y_te))
