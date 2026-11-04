import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
train = df[df["date"] < "2024-01-01"]
test = df[df["date"] >= "2024-01-01"]
