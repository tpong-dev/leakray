# expect: LK001
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier


def build(path):
    df_all = pd.read_csv(path).drop_duplicates()
    lbl = df_all["churned"]
    feats = df_all.drop(columns=["churned"])
    ct = ColumnTransformer([
        ("num", StandardScaler(), ["tenure", "monthly_spend"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
    ])
    X = ct.fit_transform(feats)
    X_tr, X_te, y_tr, y_te = train_test_split(X, lbl, test_size=0.2, random_state=0)
    model = LGBMClassifier(n_estimators=300).fit(X_tr, y_tr)
    return model.score(X_te, y_te)


print(build("telco.csv"))
