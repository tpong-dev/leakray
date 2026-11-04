# expect: LK004
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("survey.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="label"), df["label"], random_state=1)
imp = SimpleImputer(strategy="mean").fit(X_train)
X_train = imp.transform(X_train)
imp_test = SimpleImputer(strategy="mean")
imp_test.fit(X_test)
X_test = imp_test.transform(X_test)
rf = RandomForestClassifier().fit(X_train, y_train)
print(rf.score(X_test, y_test))
