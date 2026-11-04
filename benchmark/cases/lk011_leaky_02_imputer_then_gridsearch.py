# expect: LK011
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("hepatitis.csv").drop_duplicates()
X = df.drop(columns="died")
imputer = SimpleImputer(strategy="mean")
X_imp = imputer.fit_transform(X)
gs = GridSearchCV(RandomForestClassifier(), {"max_depth": [3, 5, None]}, cv=5)
gs.fit(X_imp, df["died"])
print(gs.best_score_)
