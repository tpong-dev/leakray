import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("loans.csv").drop_duplicates()
df["grade_rate"] = df["grade"].map(df.groupby("grade")["defaulted"].mean())
X_tr, X_te, y_tr, y_te = train_test_split(df[["grade_rate", "income"]], df["defaulted"])
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
