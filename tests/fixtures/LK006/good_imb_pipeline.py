import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("fraud.csv")
X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]
print(cross_val_score(make_pipeline(SMOTE(), LogisticRegression()), X, y))
