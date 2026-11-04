import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("d.csv").drop_duplicates()
X = df.drop(columns=["target"]).values
y = df["target"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = build_model()
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
