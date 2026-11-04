# expect: LK009
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

df = pd.read_csv("concrete.csv").drop_duplicates()
X = df.drop(columns="strength")
y = df["strength"]
X = X.fillna(X.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
print(SVR().fit(X_train, y_train).score(X_test, y_test))
