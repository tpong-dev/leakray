# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

scraped = (
    pd.read_csv("scraped_products.csv")
    .drop_duplicates(subset=["title", "price", "rating"])
    .reset_index(drop=True)
)
feats = scraped[["price", "rating", "num_reviews", "discount"]]
lbl = scraped["bestseller"]
X_tr, X_te, y_tr, y_te = train_test_split(feats, lbl, test_size=0.2, random_state=0)
print(RandomForestClassifier().fit(X_tr, y_tr).score(X_te, y_te))
