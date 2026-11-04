# expect: clean
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("fraud.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="fraud"), df["fraud"], stratify=df["fraud"])
X_train_res, y_train_res = SMOTE(random_state=0).fit_resample(X_train, y_train)
rf = RandomForestClassifier().fit(X_train_res, y_train_res)
print(rf.score(X_test, y_test))
