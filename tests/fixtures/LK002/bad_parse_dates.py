import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("sales.csv", parse_dates=["date"])
X = df.drop(columns=["sales"])
y = df["sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
