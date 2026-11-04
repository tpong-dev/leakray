import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
df = pd.read_csv("visits.csv")
X = df.drop(columns=["diagnosis", "patient_id"])
y = df["diagnosis"]
gkf = GroupKFold(5)
for tr, te in gkf.split(X, y, groups=df["patient_id"]):
    pass
