# expect: LK012
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

tx = pd.read_csv("card_tx.csv").drop_duplicates()
tx = tx.set_index("customer_id")
X = tx[["amount", "merchant_risk", "hour"]]
y = tx["chargeback"]
skf = StratifiedKFold(n_splits=5, shuffle=True)
print(cross_val_score(GradientBoostingClassifier(), X, y, cv=skf).mean())
