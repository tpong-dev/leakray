# expect: LK011
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

df_all = pd.read_csv("heart.csv").drop_duplicates()
feats = df_all.drop(columns="disease")
lbl = df_all["disease"]

X = StandardScaler().fit_transform(feats)
scores = cross_val_score(SVC(C=1.0), X, lbl, cv=10)
print(scores.mean())
