# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

df = pd.read_csv("autos.csv").drop_duplicates()
X_dev, X_test, y_dev, y_test = train_test_split(df[["hp", "weight"]], df["price"], test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.25)
model = make_pipeline(StandardScaler(), Ridge()).fit(X_train, y_train)
print("val", model.score(X_val, y_val))
model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
print("test", model.score(X_test, y_test))
