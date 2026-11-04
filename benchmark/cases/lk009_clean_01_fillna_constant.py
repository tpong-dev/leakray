# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("applicants.csv").drop_duplicates()
df["income"] = df["income"].fillna(0)
df["employer"] = df["employer"].fillna("unknown")
df = df.fillna(-1)
X_tr, X_te, y_tr, y_te = train_test_split(df[["income", "age"]], df["approved"])
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
