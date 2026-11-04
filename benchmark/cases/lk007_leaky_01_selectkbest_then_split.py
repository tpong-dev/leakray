# expect: LK007
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("genes.csv").drop_duplicates()
X, y = df.drop(columns="tumor"), df["tumor"]
X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.25)
print(LogisticRegression(max_iter=2000).fit(X_train, y_train).score(X_test, y_test))
