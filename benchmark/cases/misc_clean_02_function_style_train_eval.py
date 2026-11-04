# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_model(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier().fit(scaler.transform(X_train), y_train)
    return scaler, model


def evaluate(scaler, model, X_test, y_test):
    return model.score(scaler.transform(X_test), y_test)


df = pd.read_csv("wine.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns="quality"), df["quality"])
print(evaluate(*train_model(X_train, y_train), X_test, y_test))
