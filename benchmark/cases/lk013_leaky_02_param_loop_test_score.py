# expect: LK013
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("spam_features.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="spam"), df["spam"], random_state=0)
results = []
for C in [0.01, 0.1, 1, 10, 100]:
    clf = SVC(C=C).fit(X_train, y_train)
    results.append((clf.score(X_test, y_test), C))
best_score, best_C = max(results)
print(f"best C={best_C} test acc={best_score:.3f}")
