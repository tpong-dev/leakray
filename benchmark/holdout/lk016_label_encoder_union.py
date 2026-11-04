# expect: LK016
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier

data_tr = pd.read_csv("train.csv").drop_duplicates()
data_te = pd.read_csv("test.csv").drop_duplicates()

for col in ["region", "channel"]:
    le = LabelEncoder()
    le.fit(pd.concat([data_tr[col], data_te[col]]).astype(str))
    data_tr[col] = le.transform(data_tr[col].astype(str))
    data_te[col] = le.transform(data_te[col].astype(str))

stacked = pd.concat([data_tr.drop(columns="target"), data_te], ignore_index=True)
svd = TruncatedSVD(n_components=5).fit(stacked)

clf = CatBoostClassifier(verbose=0).fit(svd.transform(data_tr.drop(columns="target")), data_tr["target"])
preds = clf.predict(svd.transform(data_te))
