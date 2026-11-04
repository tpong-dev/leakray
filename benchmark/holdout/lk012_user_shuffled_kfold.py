# expect: LK012
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostClassifier


def score_sessions(sessions):
    X = sessions[["clicks", "dwell_sec", "pages", "device_mobile"]]
    y = sessions["converted"]
    folds = KFold(n_splits=5, shuffle=True, random_state=1)
    return cross_val_score(CatBoostClassifier(verbose=0), X, y, cv=folds).mean()


sessions = pd.read_csv("sessions.csv").drop_duplicates()
print(sessions["user_id"].nunique(), "users")
print(score_sessions(sessions))
