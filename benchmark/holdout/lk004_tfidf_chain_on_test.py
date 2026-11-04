# expect: LK004
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier

reviews = pd.read_csv("reviews.csv").drop_duplicates()
texts_tr, texts_te, y_tr, y_te = train_test_split(
    reviews["body"], reviews["sentiment"], test_size=0.2, random_state=2
)
vec = TfidfVectorizer(min_df=3)
tr_vec = vec.fit_transform(texts_tr)
te_vec = TfidfVectorizer(min_df=3, vocabulary=vec.vocabulary_).fit(texts_te).transform(texts_te)

model = SGDClassifier(loss="log_loss").fit(tr_vec, y_tr)
print(model.score(te_vec, y_te))
