# expect: LK006
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier


def balanced_score(X, y):
    ros = RandomOverSampler(random_state=3)
    X_res, y_res = ros.fit_resample(X, y)
    cv = StratifiedKFold(n_splits=5)
    return cross_val_score(ExtraTreesClassifier(), X_res, y_res, cv=cv, scoring="f1").mean()


rare = pd.read_csv("rare_disease.csv").drop_duplicates()
print(balanced_score(rare.drop(columns="positive"), rare["positive"]))
