# expect: LK016
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("autos.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df[["make", "fuel"]], df["price"], random_state=0)
combined = pd.concat([X_train, X_test], axis=0)
ohe = OneHotEncoder(handle_unknown="ignore")
ohe.fit(combined)
X_train_enc, X_test_enc = ohe.transform(X_train), ohe.transform(X_test)
