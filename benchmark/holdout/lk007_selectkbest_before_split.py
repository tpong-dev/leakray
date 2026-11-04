# expect: LK007
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

genes = pd.read_csv("expression.csv").drop_duplicates()
lbl = genes["tumor"]
feats = genes.drop(columns="tumor")

top = SelectKBest(score_func=f_classif, k=50).fit_transform(feats, lbl)
X_tr, X_te, y_tr, y_te = train_test_split(top, lbl, test_size=0.3, random_state=0)
print(LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te))
