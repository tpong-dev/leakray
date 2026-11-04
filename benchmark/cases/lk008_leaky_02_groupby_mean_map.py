# expect: LK008
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

df = pd.read_csv("products.csv").drop_duplicates()
cat_means = df.groupby("category")["price"].mean()
df["category_enc"] = df["category"].map(cat_means)
X = df[["category_enc", "weight", "rating"]]
X_train, X_test, y_train, y_test = train_test_split(X, df["price"], test_size=0.2)
print(Ridge().fit(X_train, y_train).score(X_test, y_test))
