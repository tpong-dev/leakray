# expect: LK015
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic_clean.csv")
X = df[["pclass", "fare", "sibsp"]]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))
