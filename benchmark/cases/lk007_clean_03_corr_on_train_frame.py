# expect: clean
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("housing.csv").drop_duplicates()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=4)
corr = train_df.corr(numeric_only=True)["median_value"].abs().drop("median_value")
keep = corr[corr > 0.3].index.tolist()
lin = LinearRegression().fit(train_df[keep], train_df["median_value"])
print(lin.score(test_df[keep], test_df["median_value"]))
