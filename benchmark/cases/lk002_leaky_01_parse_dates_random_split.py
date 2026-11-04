# expect: LK002
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("energy.csv", parse_dates=["timestamp"]).drop_duplicates()
df["hour"] = df["timestamp"].dt.hour
X = df[["hour", "temperature", "humidity"]]
y = df["consumption"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
rf = RandomForestRegressor().fit(X_train, y_train)
print(rf.score(X_test, y_test))
