# expect: LK011
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("breast.csv").drop_duplicates()
X = df.drop(columns="malignant")
y = df["malignant"]
X_scaled = StandardScaler().fit_transform(X)
scores = cross_val_score(SVC(), X_scaled, y, cv=10)
print(scores.mean())
