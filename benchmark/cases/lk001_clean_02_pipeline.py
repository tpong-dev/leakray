# expect: clean
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("clinic.csv").drop_duplicates()
X, y = df.drop(columns="outcome"), df["outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7)
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=5)),
    ("svc", SVC()),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
