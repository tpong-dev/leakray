# expect: LK006
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

df_all = pd.read_csv("fraud.csv").drop_duplicates()
feats = df_all.drop(columns="is_fraud")
lbl = df_all["is_fraud"]

feats_bal, lbl_bal = SMOTE(random_state=0).fit_resample(feats, lbl)
X_tr, X_te, y_tr, y_te = train_test_split(feats_bal, lbl_bal, test_size=0.2, random_state=0)
model = LGBMClassifier().fit(X_tr, y_tr)
print(model.score(X_te, y_te))
