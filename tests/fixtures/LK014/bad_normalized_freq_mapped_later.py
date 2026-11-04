import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("events.csv").drop_duplicates()
freq = df["browser"].value_counts(normalize=True)
df["browser_freq"] = df["browser"].map(freq)
X_tr, X_te, y_tr, y_te = train_test_split(df[["browser_freq"]], df["converted"])
