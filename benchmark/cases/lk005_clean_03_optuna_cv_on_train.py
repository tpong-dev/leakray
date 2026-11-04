# expect: clean
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

df = pd.read_csv("leads.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="converted"), df["converted"], test_size=0.2)


def objective(trial):
    clf = RandomForestClassifier(max_depth=trial.suggest_int("max_depth", 2, 16))
    return cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc").mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
final = RandomForestClassifier(**study.best_params).fit(X_train, y_train)
print(final.score(X_test, y_test))
