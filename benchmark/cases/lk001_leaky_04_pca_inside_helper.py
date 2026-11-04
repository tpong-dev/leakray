# expect: LK001
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def reduce(matrix, k=10):
    pca = PCA(n_components=k)
    return pca.fit_transform(matrix)


df = pd.read_csv("pixels.csv").drop_duplicates()
y = df.pop("digit")
X_red = reduce(df.values)
X_tr, X_te, y_tr, y_te = train_test_split(X_red, y, test_size=0.25)
knn = KNeighborsClassifier().fit(X_tr, y_tr)
print(knn.score(X_te, y_te))
