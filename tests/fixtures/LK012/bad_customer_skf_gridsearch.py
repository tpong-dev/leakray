import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

tx = pd.read_csv("tx.csv").set_index("customer_id")
X = tx[["amount", "hour"]]
y = tx["chargeback"]
skf = StratifiedKFold(n_splits=5, shuffle=True)
search = GridSearchCV(GradientBoostingClassifier(), {"max_depth": [2, 3]}, cv=skf)
search.fit(X, y)
