# expect: LK004, LK005
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

df = pd.read_csv("wine.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="quality"), df["quality"])
grid = GridSearchCV(SVC(), {"C": [0.1, 1, 10], "gamma": ["scale", 0.01]}, cv=5)
grid.fit(X_test, y_test)
best = grid.best_estimator_.fit(X_train, y_train)
print(best.score(X_test, y_test))
