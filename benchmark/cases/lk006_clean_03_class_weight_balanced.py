# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("rare_events.csv").drop_duplicates()
X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="event"), df["event"], stratify=df["event"])
rf = RandomForestClassifier(class_weight="balanced", n_estimators=300)
rf.fit(X_tr, y_tr)
print(rf.score(X_te, y_te))
