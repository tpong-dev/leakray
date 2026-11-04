# expect: clean
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

df = pd.read_csv("wine.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="quality"), df["quality"])
search = GridSearchCV(SVC(), param_grid={"C": [0.1, 1, 10]}, cv=5)
search.fit(X_train, y_train)
print("best cv", search.best_score_)
print("test", search.score(X_test, y_test))
