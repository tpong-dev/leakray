# expect: LK014
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

df_all = pd.read_csv("card_tx.csv").drop_duplicates()
df_all["merchant_freq"] = df_all["merchant"].map(df_all["merchant"].value_counts(normalize=True))

feats = df_all[["merchant_freq", "amount", "hour"]]
lbl = df_all["chargeback"]
X_tr, X_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.2, random_state=0)
print(LGBMClassifier().fit(X_tr, y_tr).score(X_te, y_te))
