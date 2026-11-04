# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

visits = pd.read_csv("visits.csv").drop_duplicates()
X = visits.drop(columns=["readmitted", "patient_id"])
y = visits["readmitted"]
scores = cross_val_score(RandomForestClassifier(), X, y, groups=visits["patient_id"], cv=GroupKFold(n_splits=5))
print(scores.mean())
