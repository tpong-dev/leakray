# expect: LK003
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("churn.csv").drop_duplicates()
features = ["tenure", "monthly_charges", "support_calls", "churn"]
X = df[features]
y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=5)
LogisticRegression(max_iter=500).fit(X_tr, y_tr)
