# expect: LK001, LK009
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("patients_vitals.csv").drop_duplicates()
df = df.fillna(df.median(numeric_only=True))
X = StandardScaler().fit_transform(df.drop(columns="sepsis"))
X_tr, X_te, y_tr, y_te = train_test_split(X, df["sepsis"], test_size=0.2)
print(LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te))
