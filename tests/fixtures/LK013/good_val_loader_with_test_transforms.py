import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

df = pd.read_csv("d.csv").drop_duplicates()
tr, va = train_test_split(df)
tf_test = make_transforms(train=False)
dl_va = make_loader(va, tf_test)
best = -1.0
for ep in range(10):
    yv, pv = predict(net, dl_va)
    f1 = f1_score(yv, pv)
    if f1 > best:
        best = f1
