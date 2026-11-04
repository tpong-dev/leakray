# expect: clean
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

df = pd.read_csv("bikes.csv").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(df[["temp", "wind"]], df["rentals"])
pipe = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
print(mean_absolute_error(y_test, preds))
