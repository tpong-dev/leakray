# expect: LK008
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("loans.csv").drop_duplicates()
te = TargetEncoder(cols=["region", "employer_type"])
X = te.fit_transform(df[["region", "employer_type", "income"]], df["defaulted"])
X_train, X_test, y_train, y_test = train_test_split(X, df["defaulted"], stratify=df["defaulted"])
print(LogisticRegression().fit(X_train, y_train).score(X_test, y_test))
