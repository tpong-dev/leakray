import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("claims.csv")
train, test = train_test_split(df, stratify=df["approved"])
minority = train[train["approved"] == 1]
minority_up = resample(minority, replace=True, n_samples=1000)
train = pd.concat([train, minority_up])
