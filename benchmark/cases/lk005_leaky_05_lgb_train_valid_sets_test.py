# expect: LK005
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("ctr.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="click"), df["click"], test_size=0.2)
dtrain = lgb.Dataset(X_train, label=y_train)
dtest = lgb.Dataset(X_test, label=y_test)
booster = lgb.train({"objective": "binary"}, dtrain, num_boost_round=1000,
                    valid_sets=[dtest], callbacks=[lgb.early_stopping(50)])
print(booster.best_iteration)
