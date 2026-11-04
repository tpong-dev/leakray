# expect: clean
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

counts = pd.read_csv("word_counts.csv").drop_duplicates()
X, y = counts.drop(columns="spam"), counts["spam"]
pipe = make_pipeline(SelectKBest(chi2, k=200), MultinomialNB())
print(cross_val_score(pipe, X, y, cv=5).mean())
