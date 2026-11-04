# expect: LK014
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("listings.csv").drop_duplicates()
df["city_freq"] = df["city"].map(df["city"].value_counts())
X = df[["city_freq", "price", "reviews"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, df["booked"], random_state=0)
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
