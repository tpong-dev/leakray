# expect: LK007
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("housing.csv").drop_duplicates()
corr = df.corr(numeric_only=True)["median_value"].abs().drop("median_value")
keep = corr[corr > 0.3].index.tolist()
X_train, X_test, y_train, y_test = train_test_split(df[keep], df["median_value"], test_size=0.2)
lin = LinearRegression().fit(X_train, y_train)
print(lin.score(X_test, y_test))
