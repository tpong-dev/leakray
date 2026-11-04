import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("claims.csv")
majority = df[df["approved"] == 0]
minority = df[df["approved"] == 1]
minority_up = resample(minority, replace=True, n_samples=len(majority))
balanced = pd.concat([majority, minority_up])
X_train, X_test, y_train, y_test = train_test_split(balanced.drop(columns="approved"), balanced["approved"])
