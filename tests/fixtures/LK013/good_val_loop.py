import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)
scores = {}
for depth in range(1, 20):
    clf = DecisionTreeClassifier(max_depth=depth).fit(X_tr, y_tr)
    scores[depth] = clf.score(X_val, y_val)
best_depth = max(scores, key=scores.get)
final = DecisionTreeClassifier(max_depth=best_depth).fit(X_train, y_train)
print(final.score(X_test, y_test))
