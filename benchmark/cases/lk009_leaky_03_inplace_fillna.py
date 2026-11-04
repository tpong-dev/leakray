# expect: LK009
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

data = pd.read_csv("air_quality.csv")
data.drop_duplicates(inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)
features = data.drop(columns=["pm25"])
X_train, X_test, y_train, y_test = train_test_split(features, data["pm25"], test_size=0.3)
print(KNeighborsRegressor().fit(X_train, y_train).score(X_test, y_test))
