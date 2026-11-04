# expect: LK009
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

df_all = pd.read_csv("energy.csv").drop_duplicates()
df_all = df_all.fillna(df_all.mean(numeric_only=True))

feats, lbl = df_all.drop(columns="kwh"), df_all["kwh"]
X_tr, X_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.2, random_state=0)
print(Ridge(alpha=1.0).fit(X_tr, y_tr).score(X_te, y_te))
