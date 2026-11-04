# expect: LK001
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv("clinic.csv").drop_duplicates()
cols = ["bmi", "glucose", "bp", "insulin"]
X_imp = SimpleImputer(strategy="median").fit_transform(data[cols])
X_train, X_test, y_train, y_test = train_test_split(X_imp, data["outcome"], stratify=data["outcome"])
model = SVC().fit(X_train, y_train)
print("acc", model.score(X_test, y_test))
