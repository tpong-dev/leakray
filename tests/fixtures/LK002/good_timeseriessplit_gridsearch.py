import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
X = df[["promo", "price"]]
y = df["units"]
search = GridSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=TimeSeriesSplit(n_splits=4))
search.fit(X, y)
