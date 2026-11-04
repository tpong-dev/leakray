# expect: LK001
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("bank.csv").drop_duplicates()
X = df.drop(columns=["default"])
y = df["default"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(clf.score(X_test, y_test))
