# expect: LK002
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from catboost import CatBoostClassifier


def evaluate(frame):
    frame = frame.drop_duplicates()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    X = frame[["sensor_a", "sensor_b", "load"]]
    y = frame["failure"]
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    return cross_val_score(CatBoostClassifier(verbose=0), X, y, cv=cv).mean()


log = pd.read_csv("machine_log.csv")
print(evaluate(log))
