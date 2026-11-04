# expect: LK009
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("weather_stations.csv").drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))
X_train, X_test, y_train, y_test = train_test_split(df[["pressure", "humidity"]], df["rain_mm"])
print(LinearRegression().fit(X_train, y_train).score(X_test, y_test))
