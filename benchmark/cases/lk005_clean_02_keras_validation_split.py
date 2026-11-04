# expect: clean
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

data = pd.read_parquet("signals.parquet").drop_duplicates()
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns="label"), data["label"], test_size=0.2, random_state=0)
net = keras.Sequential([keras.layers.Dense(32, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
net.compile(optimizer="adam", loss="binary_crossentropy")
net.fit(X_train, y_train, epochs=30, validation_split=0.2)
print(net.evaluate(X_test, y_test))
