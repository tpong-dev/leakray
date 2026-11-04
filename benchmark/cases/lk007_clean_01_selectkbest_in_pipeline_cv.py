# expect: clean
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv("genes.csv").drop_duplicates()
X, y = df.drop(columns="tumor"), df["tumor"]
pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=2000))
print(cross_val_score(pipe, X, y, cv=5).mean())
