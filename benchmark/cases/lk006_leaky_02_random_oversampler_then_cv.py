# expect: LK006
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("defects.csv").drop_duplicates()
X = df.drop(columns="defect")
y = df["defect"]
ros = RandomOverSampler()
X, y = ros.fit_resample(X, y)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv, scoring="f1").mean())
