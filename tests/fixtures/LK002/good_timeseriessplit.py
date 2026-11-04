import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
X = df.drop(columns=["sales"])
tscv = TimeSeriesSplit(n_splits=5)
for tr, te in tscv.split(X):
    print(tr, te)
