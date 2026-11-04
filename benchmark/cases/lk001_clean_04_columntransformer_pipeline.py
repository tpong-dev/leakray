# expect: clean
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("telco.csv").drop_duplicates()
pre = ColumnTransformer([
    ("num", StandardScaler(), ["tenure", "monthly_charges"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract", "internet"]),
])
model = make_pipeline(pre, GradientBoostingClassifier())
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="churn"), df["churn"], test_size=0.2)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
