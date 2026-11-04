import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("sales.csv").sort_values(["store", "date"])
y = df["sales"]
df["roll"] = df.groupby("store")["sales"].rolling(28).mean().reset_index(level=0, drop=True)
