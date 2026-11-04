# expect: clean
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("adult.csv").drop_duplicates()
X, y = df.drop(columns="income_gt_50k"), df["income_gt_50k"]
pre = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler()), ["age", "hours_per_week"]),
    (OneHotEncoder(handle_unknown="ignore"), ["workclass", "education"]),
)
search = GridSearchCV(make_pipeline(pre, LogisticRegression(max_iter=1000)), {"logisticregression__C": [0.1, 1, 10]})
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
search.fit(X_train, y_train)
print(search.score(X_test, y_test))
