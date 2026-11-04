# expect: clean
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from lightgbm import LGBMClassifier

visits = pd.read_csv("ecg_visits.csv").drop_duplicates()
feats = visits.drop(columns=["patient_id", "arrhythmia"])
lbl = visits["arrhythmia"]
groups = visits["patient_id"]
cv = GroupKFold(n_splits=5)
print(cross_val_score(LGBMClassifier(), feats, lbl, cv=cv, groups=groups).mean())
