# expect: LK005
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

df_all = pd.read_csv("credit.csv").drop_duplicates()
feats, lbl = df_all.drop(columns="bad_loan"), df_all["bad_loan"]
X_train, X_test, y_train, y_test = train_test_split(feats, lbl, test_size=0.2, random_state=0)

gbm = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03)
gbm.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(100)],
)
print(gbm.score(X_test, y_test))
