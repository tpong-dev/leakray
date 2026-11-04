import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv("sales.csv", parse_dates=["date"]).sort_values("date")
y = df["sales"]
df["temp_7d"] = df["temperature"].rolling(7).mean()
