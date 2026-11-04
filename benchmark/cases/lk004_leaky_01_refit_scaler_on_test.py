# expect: LK004
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("heart.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="disease"), df["disease"])
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
model = LogisticRegression().fit(X_train, y_train)
print(model.score(X_test, y_test))
