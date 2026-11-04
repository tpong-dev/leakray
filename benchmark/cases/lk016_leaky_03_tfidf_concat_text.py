# expect: LK016
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("train_tweets.csv").drop_duplicates()
test = pd.read_csv("test_tweets.csv")
all_text = pd.concat([train.text, test.text], ignore_index=True)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
tfidf.fit(all_text)
clf = LogisticRegression(max_iter=1000).fit(tfidf.transform(train.text), train.target)
preds = clf.predict(tfidf.transform(test.text))
