# expect: clean
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("subscribers.csv").drop_duplicates()
df["spend_per_month"] = df["total_spend"] / df["tenure"].clip(lower=1)
df["log_spend"] = np.log1p(df["monthly_spend"])
X_tr, X_te, y_tr, y_te = train_test_split(df[["spend_per_month", "log_spend"]], df["churn"])
print(LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te))
