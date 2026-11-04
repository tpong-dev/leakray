import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
df = pd.read_csv("energy.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
X = df.drop(columns=["load"])
y = df["load"]
kf = KFold(n_splits=5, shuffle=True)
for tr, te in kf.split(X):
    print(len(tr), len(te))
