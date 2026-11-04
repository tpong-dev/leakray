# expect: clean
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

tx = pd.read_csv("card_tx.csv").drop_duplicates()
X = tx[["amount", "merchant_risk"]]
y = tx["chargeback"]
cv = StratifiedGroupKFold(n_splits=5)
print(cross_val_score(GradientBoostingClassifier(), X, y, cv=cv, groups=tx["customer_id"]).mean())
