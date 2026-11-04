# expect: clean
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

msgs = pd.read_csv("sms.csv").drop_duplicates(subset=["message"])
m_train, m_test, y_train, y_test = train_test_split(msgs["message"], msgs["is_spam"], test_size=0.2)
vec = CountVectorizer()
nb = MultinomialNB().fit(vec.fit_transform(m_train), y_train)
print(nb.score(vec.transform(m_test), y_test))
