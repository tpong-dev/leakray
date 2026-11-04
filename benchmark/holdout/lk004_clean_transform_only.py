# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier

cars = pd.read_csv("cars.csv").drop_duplicates()
cat_cols = ["make", "fuel", "gearbox"]
X = cars.drop(columns="sold_fast")
y = cars["sold_fast"]
test_size = 0.25
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=8)

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X_tr[cat_cols])
X_tr[cat_cols] = enc.transform(X_tr[cat_cols])
X_te[cat_cols] = enc.transform(X_te[cat_cols])

model = CatBoostClassifier(verbose=0).fit(X_tr, y_tr)
preds = model.predict(X_te)
print((preds == y_te).mean())
