# expect: LK009
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("applicants.csv").drop_duplicates()
df["income"] = df["income"].fillna(df["income"].median())
df["age"] = df["age"].fillna(df["age"].mean())
X = df[["income", "age", "years_employed"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, df["approved"], random_state=11)
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
