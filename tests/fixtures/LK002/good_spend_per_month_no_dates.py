import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("subscribers.csv")
df["spend_per_month"] = df["total_spend"] / df["tenure"].clip(lower=1)
df["visits_per_week"] = df["visits"] / 4
df["hour_of_day"] = df["hour_of_day"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(df[["spend_per_month", "visits_per_week", "hour_of_day"]], df["churn"])
print(LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te))
