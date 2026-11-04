import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10]})
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
