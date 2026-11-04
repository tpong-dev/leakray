import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("data.csv").drop_duplicates()
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
