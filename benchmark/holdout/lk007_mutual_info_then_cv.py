# expect: LK007
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier

data = pd.read_csv("sensors_wide.csv").drop_duplicates()
y = data["defect"].to_numpy()
X = data.drop(columns="defect").to_numpy()

mi = mutual_info_classif(X, y, random_state=0)
keep = np.argsort(mi)[::-1][:20]
X_sel = X[:, keep]
print(cross_val_score(LGBMClassifier(), X_sel, y, cv=5).mean())
