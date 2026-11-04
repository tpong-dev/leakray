# expect: clean
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

reviews = pd.read_csv("reviews.csv").drop_duplicates(subset=["text"])
txt_train, txt_test, y_train, y_test = train_test_split(reviews["text"], reviews["stars"], test_size=0.2)
vec = TfidfVectorizer(min_df=2)
X_train = vec.fit_transform(txt_train)
X_test = vec.transform(txt_test)
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print(clf.score(X_test, y_test))
