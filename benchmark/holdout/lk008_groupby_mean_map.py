# expect: LK008
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

df_all = pd.read_csv("listings.csv").drop_duplicates()
brand_rate = df_all.groupby("brand")["sold"].mean()
df_all["brand_rate"] = df_all["brand"].map(brand_rate)

feats = df_all[["brand_rate", "price", "photos"]]
lbl = df_all["sold"]
X_tr, X_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.2, random_state=0)
print(LGBMClassifier().fit(X_tr, y_tr).score(X_te, y_te))
