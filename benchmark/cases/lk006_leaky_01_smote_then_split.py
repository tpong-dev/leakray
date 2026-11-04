# expect: LK006
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("fraud.csv").drop_duplicates()
X, y = df.drop(columns="fraud"), df["fraud"]
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
rf = RandomForestClassifier().fit(X_train, y_train)
print(rf.score(X_test, y_test))
