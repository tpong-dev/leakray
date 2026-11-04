# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

df_all = pd.read_csv("card_tx.csv").drop_duplicates()
data_tr, data_te = train_test_split(df_all, test_size=0.2, random_state=0)

freq = data_tr["merchant"].value_counts(normalize=True)
data_tr = data_tr.assign(merchant_freq=data_tr["merchant"].map(freq))
data_te = data_te.assign(merchant_freq=data_te["merchant"].map(freq).fillna(0.0))

cols = ["merchant_freq", "amount", "hour"]
clf = LGBMClassifier().fit(data_tr[cols], data_tr["chargeback"])
print(clf.score(data_te[cols], data_te["chargeback"]))
