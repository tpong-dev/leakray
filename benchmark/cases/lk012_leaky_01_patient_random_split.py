# expect: LK012
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

visits = pd.read_csv("visits.csv").drop_duplicates()
X = visits.drop(columns=["readmitted", "patient_id"])
y = visits["readmitted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(RandomForestClassifier().fit(X_train, y_train).score(X_test, y_test))
