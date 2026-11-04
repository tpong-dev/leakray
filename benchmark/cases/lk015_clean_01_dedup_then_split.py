# expect: clean
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic_clean.csv")
df = df.drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df[["pclass", "fare"]], df["survived"], random_state=42)
print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))
