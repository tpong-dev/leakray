# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("diabetes.csv").drop_duplicates()
X_tmp, X_test, y_tmp, y_test = train_test_split(df.drop(columns="outcome"), df["outcome"], test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25)
models = {"lr": LogisticRegression(max_iter=1000), "rf": RandomForestClassifier()}
val_scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    val_scores[name] = model.score(X_val, y_val)
best = models[max(val_scores, key=val_scores.get)]
print("test", best.score(X_test, y_test))
