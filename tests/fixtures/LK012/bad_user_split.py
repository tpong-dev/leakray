import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("events.csv").drop_duplicates()
X = df.drop(columns=["clicked"])
y = df["clicked"]
users = df["user_id"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
