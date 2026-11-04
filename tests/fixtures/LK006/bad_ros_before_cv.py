import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("fraud.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
ros = RandomOverSampler()
X_r, y_r = ros.fit_resample(X, y)
print(cross_val_score(RandomForestClassifier(), X_r, y_r, cv=5))
