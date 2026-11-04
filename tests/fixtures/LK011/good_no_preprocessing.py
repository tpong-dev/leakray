import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("d.csv")
X = df.drop(columns=["target"])
y = df["target"]
print(cross_val_score(RandomForestClassifier(), X, y))
