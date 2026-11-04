# expect: LK007
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sensors.csv").drop_duplicates()
X = df.drop(columns="failure")
y = df["failure"]
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=8)
rfe.fit(X, y)
X_small = X.loc[:, rfe.support_]
print(cross_val_score(SVC(), X_small, y, cv=5).mean())
