import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

events = pd.read_csv("sessions.csv")
X = events[["clicks", "pages"]]
y = events["purchased"]
print(cross_val_score(LogisticRegression(), X, y, cv=GroupKFold(5), groups=events["user_id"]).mean())
