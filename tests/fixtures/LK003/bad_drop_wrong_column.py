import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("churn.csv")
y = df["churn"]
X = df.drop(columns=["customer_name"])
X_train, X_test, y_train, y_test = train_test_split(X, y)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
