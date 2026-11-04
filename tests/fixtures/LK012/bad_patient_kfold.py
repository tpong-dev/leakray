import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
df = pd.read_csv("visits.csv")
X = df.drop(columns=["diagnosis", "patient_id"])
y = df["diagnosis"]
skf = StratifiedKFold(5, shuffle=True)
for tr, te in skf.split(X, y):
    pass
