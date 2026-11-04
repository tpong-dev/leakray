import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("houses.csv")
model = LinearRegression()
model.fit(df, df["price"])
