# expect: clean
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("products.csv").drop_duplicates()
train, test = train_test_split(df, test_size=0.2, random_state=0)
cat_means = train.groupby("category")["price"].mean()
global_mean = train["price"].mean()
for part in (train, test):
    part["category_enc"] = part["category"].map(cat_means).fillna(global_mean)
cols = ["category_enc", "weight"]
print(Ridge().fit(train[cols], train["price"]).score(test[cols], test["price"]))
