# expect: clean
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("marketing.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="responded"), df["responded"])
selector = SelectKBest(mutual_info_classif, k=10)
X_train_s = selector.fit_transform(X_train, y_train)
X_test_s = selector.transform(X_test)
print(RandomForestClassifier().fit(X_train_s, y_train).score(X_test_s, y_test))
