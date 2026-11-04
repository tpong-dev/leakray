import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("weather.csv")
train, test = train_test_split(df, test_size=0.2)
fill = train.mean(numeric_only=True)
train, test = train.fillna(fill), test.fillna(fill)
LinearRegression().fit(train[["p"]], train["rain"]).score(test[["p"]], test["rain"])
