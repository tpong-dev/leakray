# expect: clean
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("subscribers.csv").drop_duplicates()
tr, te = train_test_split(df, test_size=0.2, random_state=1)
region_avg = tr.groupby("region")["monthly_spend"].mean()
tr = tr.assign(region_avg=tr["region"].map(region_avg))
te = te.assign(region_avg=te["region"].map(region_avg))
cols = ["region_avg", "tenure"]
print(LogisticRegression().fit(tr[cols], tr["churn"]).score(te[cols].fillna(0), te["churn"]))
