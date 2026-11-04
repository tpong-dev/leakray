# expect: LK005
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("leads.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="converted"), df["converted"], test_size=0.2)


def objective(trial):
    clf = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        max_depth=trial.suggest_int("max_depth", 2, 16),
    )
    clf.fit(X_train, y_train)
    return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_value)
