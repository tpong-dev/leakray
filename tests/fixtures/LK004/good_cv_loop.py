import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
kf = KFold(5, shuffle=True, random_state=0)
for tr_idx, te_idx in kf.split(X):
    m = Ridge()
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    print(m.score(X.iloc[te_idx], y.iloc[te_idx]))
