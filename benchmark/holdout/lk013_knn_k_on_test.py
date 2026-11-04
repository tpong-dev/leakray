# expect: LK013
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

seeds = pd.read_csv("seeds.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(
    seeds.drop(columns="variety"), seeds["variety"], test_size=0.3, random_state=0
)

best_k, best_acc = None, 0.0
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    if acc > best_acc:
        best_k, best_acc = k, acc
print(f"best k={best_k} test acc={best_acc:.3f}")
