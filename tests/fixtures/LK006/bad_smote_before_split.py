import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
df = pd.read_csv("fraud.csv").drop_duplicates()
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
