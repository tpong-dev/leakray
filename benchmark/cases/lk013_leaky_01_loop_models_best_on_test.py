# expect: LK013
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("diabetes.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="outcome"), df["outcome"])
models = {"lr": LogisticRegression(max_iter=1000), "rf": RandomForestClassifier(), "svc": SVC()}
best_name, best_acc = None, 0.0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    if acc > best_acc:
        best_name, best_acc = name, acc
print("winner", best_name, best_acc)
