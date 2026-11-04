# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

seeds = pd.read_csv("seeds.csv").drop_duplicates()
X_rest, X_test, y_rest, y_test = train_test_split(
    seeds.drop(columns="variety"), seeds["variety"], test_size=0.2, random_state=0
)
X_fit, X_val, y_fit, y_val = train_test_split(X_rest, y_rest, test_size=0.25, random_state=1)

val_scores = {}
for k in range(1, 31):
    val_scores[k] = KNeighborsClassifier(n_neighbors=k).fit(X_fit, y_fit).score(X_val, y_val)
best_k = max(val_scores, key=val_scores.get)

final = KNeighborsClassifier(n_neighbors=best_k).fit(X_rest, y_rest)
print("held-out accuracy:", final.score(X_test, y_test))
