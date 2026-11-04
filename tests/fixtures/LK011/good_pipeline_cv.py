import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
df = pd.read_csv("d.csv")
X = df.drop(columns=["target"])
y = df["target"]
print(cross_val_score(make_pipeline(StandardScaler(), SVC()), X, y, cv=5).mean())
