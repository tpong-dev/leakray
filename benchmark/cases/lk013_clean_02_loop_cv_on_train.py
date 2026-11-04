# expect: clean
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC

df = pd.read_csv("spam_features.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="spam"), df["spam"], random_state=0)
cv_means = {}
for C in [0.01, 0.1, 1, 10]:
    cv_means[C] = cross_val_score(SVC(C=C), X_train, y_train, cv=5).mean()
best_C = max(cv_means, key=cv_means.get)
final = SVC(C=best_C).fit(X_train, y_train)
print("held-out", final.score(X_test, y_test))
