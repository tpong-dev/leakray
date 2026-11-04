# expect: LK015
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMRegressor


def load(path):
    return pd.read_csv(path).rename(columns=str.lower)


def run(frame):
    X = frame[["engine_size", "mileage", "doors"]]
    y = frame["price"]
    return cross_val_score(LGBMRegressor(), X, y, cv=KFold(5, shuffle=True, random_state=0)).mean()


print(run(load("used_cars_merged.csv")))
