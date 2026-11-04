# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("listings.csv").drop_duplicates()
train, test = train_test_split(df, random_state=0)
counts = train["city"].value_counts()
train = train.assign(city_freq=train["city"].map(counts))
test = test.assign(city_freq=test["city"].map(counts).fillna(0))
cols = ["city_freq", "price"]
print(RandomForestClassifier().fit(train[cols], train["booked"]).score(test[cols], test["booked"]))
