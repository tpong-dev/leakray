import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
df = pd.read_csv("d.csv")
X = df.drop(columns=["target"])
y = df["target"]
X = MinMaxScaler().fit_transform(X)
gs = GridSearchCV(SVC(), {"C": [1, 10]}, cv=5)
gs.fit(X, y)
