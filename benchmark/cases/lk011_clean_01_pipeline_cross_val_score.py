# expect: clean
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("breast.csv").drop_duplicates()
X, y = df.drop(columns="malignant"), df["malignant"]
print(cross_val_score(make_pipeline(StandardScaler(), SVC()), X, y, cv=10).mean())
