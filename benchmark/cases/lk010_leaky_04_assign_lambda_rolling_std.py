# expect: LK010
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

web = (
    pd.read_csv("web_traffic.csv")
    .drop_duplicates()
    .assign(visits_std3=lambda d: d["visits"].rolling(3).std())
    .dropna()
)
y = web["visits"]
X_tr, X_te, y_tr, y_te = train_test_split(web[["visits_std3", "ads_spend"]], y, shuffle=False)
print(LinearRegression().fit(X_tr, y_tr).score(X_te, y_te))
