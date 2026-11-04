# expect: clean
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv")
feats = ["f1", "f2", "f3"]
rf = RandomForestClassifier().fit(train[feats], train["label"])
preds = pd.Series(rf.predict(test[feats]), name="label")
submission = pd.concat([test[["name"]].reset_index(drop=True), preds], axis=1)
submission.to_csv("submission.csv", index=False)
