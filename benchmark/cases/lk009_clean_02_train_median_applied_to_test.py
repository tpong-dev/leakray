# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df = pd.read_csv("concrete.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="strength"), df["strength"])
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
print(SVR().fit(X_train, y_train).score(X_test, y_test))
