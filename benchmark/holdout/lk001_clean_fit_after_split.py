# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor

data = pd.read_parquet("houses.parquet").drop_duplicates()
y = data.pop("price")
data_tr, data_te, y_tr, y_te = train_test_split(data, y, test_size=0.2, random_state=1)

imp = SimpleImputer(strategy="median").fit(data_tr)
scaler = MinMaxScaler().fit(imp.transform(data_tr))
Xtr = scaler.transform(imp.transform(data_tr))
Xte = scaler.transform(imp.transform(data_te))

model = CatBoostRegressor(verbose=0).fit(Xtr, y_tr)
print(model.score(Xte, y_te))
