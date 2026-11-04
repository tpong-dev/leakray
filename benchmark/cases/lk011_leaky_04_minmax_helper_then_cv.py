# expect: LK011
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def normalise(frame):
    return MinMaxScaler().fit_transform(frame)


df = pd.read_csv("seeds.csv").drop_duplicates()
X = normalise(df.drop(columns="variety"))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
print(cross_val_score(KNeighborsClassifier(), X, df["variety"], cv=skf).mean())
