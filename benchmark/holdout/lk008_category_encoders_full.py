# expect: LK008
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

rent = pd.read_csv("rentals.csv").drop_duplicates()
lbl = rent["monthly_rent"]
feats = rent.drop(columns="monthly_rent")

te = ce.TargetEncoder(cols=["neighbourhood", "building_type"])
feats_enc = te.fit_transform(feats, lbl)
data_tr, data_te, y_tr, y_te = train_test_split(feats_enc, lbl, test_size=0.2, random_state=9)
print(CatBoostRegressor(verbose=0).fit(data_tr, y_tr).score(data_te, y_te))
