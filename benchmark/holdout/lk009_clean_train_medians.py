# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

df_all = pd.read_csv("energy.csv").drop_duplicates()
df_all["notes_len"] = df_all["notes_len"].fillna(0)
feats, lbl = df_all.drop(columns="kwh"), df_all["kwh"]
data_tr, data_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.2, random_state=0)

medians = data_tr.median(numeric_only=True)
data_tr = data_tr.fillna(medians)
data_te = data_te.fillna(medians)
print(Ridge().fit(data_tr, y_tr).score(data_te, y_te))
