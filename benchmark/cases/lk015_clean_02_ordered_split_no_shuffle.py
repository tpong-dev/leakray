# expect: clean
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("sensor_stream.csv")
X = df[["s1", "s2", "s3"]]
y = df["target_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(Ridge().fit(X_train, y_train).score(X_test, y_test))
