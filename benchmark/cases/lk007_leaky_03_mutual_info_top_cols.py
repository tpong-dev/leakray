# expect: LK007
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("marketing.csv").drop_duplicates()
X = df.drop(columns="responded")
y = df["responded"]
mi = pd.Series(mutual_info_classif(X, y), index=X.columns)
top = mi.sort_values(ascending=False).head(10).index
X_train, X_test, y_train, y_test = train_test_split(X[top], y, random_state=0)
print(RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test))
