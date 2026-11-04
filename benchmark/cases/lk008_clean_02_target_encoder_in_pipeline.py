# expect: clean
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder

df = pd.read_csv("loans.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df[["region", "employer_type"]], df["defaulted"])
pipe = make_pipeline(TargetEncoder(), LogisticRegression())
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
