# expect: LK014
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("subscribers.csv").drop_duplicates()
df["region_avg_spend"] = df.groupby("region")["monthly_spend"].transform("mean")
df["spend_vs_region"] = df["monthly_spend"] / df["region_avg_spend"]
X = df[["spend_vs_region", "tenure"]]
X_tr, X_te, y_tr, y_te = train_test_split(X, df["churn"], test_size=0.2)
print(LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te))
