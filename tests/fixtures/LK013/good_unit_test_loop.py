import pandas as pd

train = pd.read_csv("train.csv").drop_duplicates()
test = pd.read_csv("test.csv").drop_duplicates()
best_model = None
for cls in (ModelA, ModelB):
    pred = cls().fit(train).predict(test)
    assert len(pred) == len(test)
