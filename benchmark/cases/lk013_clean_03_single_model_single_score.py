# expect: clean
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("yield.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="tons"), df["tons"])
gbr = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05).fit(X_train, y_train)
print("R2:", r2_score(y_test, gbr.predict(X_test)))
