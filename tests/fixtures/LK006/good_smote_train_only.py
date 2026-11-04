import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
df = pd.read_csv("fraud.csv").drop_duplicates()
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
