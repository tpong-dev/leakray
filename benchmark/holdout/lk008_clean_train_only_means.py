# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

df_all = pd.read_csv("listings.csv").drop_duplicates()
data_tr, data_te = train_test_split(df_all, test_size=0.2, random_state=0)

rate = data_tr.groupby("brand")["sold"].mean()
prior = data_tr["sold"].mean()
data_tr = data_tr.assign(brand_rate=data_tr["brand"].map(rate))
data_te = data_te.assign(brand_rate=data_te["brand"].map(rate).fillna(prior))

cols = ["brand_rate", "price", "photos"]
clf = HistGradientBoostingClassifier().fit(data_tr[cols], data_tr["sold"])
print(clf.score(data_te[cols], data_te["sold"]))
