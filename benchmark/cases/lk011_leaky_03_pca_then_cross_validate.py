# expect: LK011
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

faces = pd.read_csv("faces.csv").drop_duplicates()
y = faces.pop("person")
X_pca = PCA(n_components=50, whiten=True).fit_transform(faces)
cv_res = cross_validate(LogisticRegression(max_iter=2000), X_pca, y, cv=5, scoring="accuracy")
print(cv_res["test_score"].mean())
