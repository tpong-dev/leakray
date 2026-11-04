import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import resample

df = pd.read_csv("loans.csv")
df = resample(df, replace=True, n_samples=2000, random_state=0)
X = df[["income", "dwell_time", "loan_term_months", "years_employed"]]
y = df["default"]
print(cross_val_score(RandomForestClassifier(), X, y, cv=KFold(5, shuffle=True, random_state=0)).mean())
