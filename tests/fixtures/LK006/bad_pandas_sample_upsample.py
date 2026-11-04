import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("claims.csv")
minority = df[df["label"] == 1]
up = minority.sample(n=5000, replace=True, random_state=0)
balanced = pd.concat([df, up])
train, test = train_test_split(balanced)
