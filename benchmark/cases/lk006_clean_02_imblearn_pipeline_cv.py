# expect: clean
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("defects.csv").drop_duplicates()
X, y = df.drop(columns="defect"), df["defect"]
pipe = Pipeline([("smote", SMOTE()), ("lr", LogisticRegression(max_iter=1000))])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean())
