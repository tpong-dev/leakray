# expect: clean
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

events = pd.read_csv("sessions.csv").drop_duplicates()
X = events[["clicks", "dwell_time"]]
y = events["purchased"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups=events["user_id"]))
model = LogisticRegression().fit(X.iloc[train_idx], y.iloc[train_idx])
print(model.score(X.iloc[test_idx], y.iloc[test_idx]))
