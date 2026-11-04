import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("churn.csv").drop_duplicates()
print(df.groupby("plan")["churn"].mean())
X_tr, X_te, y_tr, y_te = train_test_split(df[["tenure", "spend"]], df["churn"])
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
