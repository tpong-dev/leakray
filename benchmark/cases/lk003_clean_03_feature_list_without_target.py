# expect: clean
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("churn.csv").drop_duplicates()
features = ["tenure", "monthly_charges", "support_calls"]
X_tr, X_te, y_tr, y_te = train_test_split(df[features], df["churn"], test_size=0.2)
model = LogisticRegression(max_iter=500)
model.fit(X_tr, y_tr)
print(model.score(X_te, y_te))
