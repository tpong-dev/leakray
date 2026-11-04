# expect: LK003
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris_extended.csv").drop_duplicates()
train_df, test_df = train_test_split(df, test_size=0.3, random_state=2)
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(train_df, train_df["species"])
print(tree.score(test_df, test_df["species"]))
