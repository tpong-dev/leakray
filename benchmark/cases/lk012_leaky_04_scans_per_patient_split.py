# expect: LK012
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

meta = pd.read_csv("xray_meta.csv").drop_duplicates()
feats = np.load("xray_features.npy")
labels = meta["pneumonia"].values
groups = meta["patient_id"].values
X_tr, X_te, y_tr, y_te = train_test_split(feats, labels, test_size=0.2, stratify=labels)
print(SVC().fit(X_tr, y_tr).score(X_te, y_te), len(set(groups)))
