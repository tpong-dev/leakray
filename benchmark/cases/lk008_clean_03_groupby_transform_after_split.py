# expect: clean
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("rentals.csv").drop_duplicates()
train_df, test_df = train_test_split(df, random_state=0)
train_df = train_df.copy()
train_df["city_te"] = train_df.groupby("city")["rent"].transform("mean")
mapping = train_df.groupby("city")["city_te"].first()
test_df = test_df.assign(city_te=test_df["city"].map(mapping))
cols = ["city_te", "sqft"]
print(GradientBoostingRegressor().fit(train_df[cols], train_df["rent"]).score(test_df[cols].fillna(0), test_df["rent"]))
