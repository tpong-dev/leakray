# expect: LK012
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

events = pd.read_csv("sessions.csv").drop_duplicates()
print(events["user_id"].nunique(), "users")
X = events[["clicks", "dwell_time", "pages"]]
y = events["purchased"]
print(cross_val_score(LogisticRegression(), X, y, cv=KFold(5, shuffle=True, random_state=0)).mean())
