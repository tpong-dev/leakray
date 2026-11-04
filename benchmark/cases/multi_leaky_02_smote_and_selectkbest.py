# expect: LK006, LK007
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("word_counts.csv").drop_duplicates()
X, y = df.drop(columns="topic"), df["topic"]
X, y = SMOTE().fit_resample(X, y)
X = SelectKBest(chi2, k=200).fit_transform(X, y)
X_tr, X_te, y_tr, y_te = train_test_split(X, y)
print(MultinomialNB().fit(X_tr, y_tr).score(X_te, y_te))
