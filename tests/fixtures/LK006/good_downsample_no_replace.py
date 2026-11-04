import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv("claims.csv")
majority = df[df["approved"] == 0]
majority_down = resample(majority, replace=False, n_samples=100)
train, test = train_test_split(majority_down)
