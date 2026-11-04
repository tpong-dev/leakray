# expect: LK004
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df_all = pd.read_csv("wine.csv").drop_duplicates()
feats = df_all.drop(columns="quality")
lbl = df_all["quality"]
data_tr, data_te, lbl_tr, lbl_te = train_test_split(feats, lbl, test_size=0.2, random_state=5)

sc = StandardScaler()
Xtr = sc.fit_transform(data_tr)
Xte = sc.fit_transform(data_te)

knn = KNeighborsClassifier(n_neighbors=7).fit(Xtr, lbl_tr)
print(knn.score(Xte, lbl_te))
