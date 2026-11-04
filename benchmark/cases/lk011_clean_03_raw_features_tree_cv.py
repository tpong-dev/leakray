# expect: clean
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("seeds.csv").drop_duplicates()
X = df.drop(columns="variety")
print(cross_val_score(ExtraTreesClassifier(n_estimators=300), X, df["variety"], cv=5).mean())
