# expect: clean
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

churn = pd.read_csv("churn.csv").drop_duplicates()
X, y = churn.drop(columns="left"), churn["left"]
pipe = Pipeline([
    ("smote", SMOTE(random_state=0)),
    ("rf", RandomForestClassifier(n_estimators=300)),
])
print(cross_val_score(pipe, X, y, cv=StratifiedKFold(5), scoring="roc_auc").mean())
