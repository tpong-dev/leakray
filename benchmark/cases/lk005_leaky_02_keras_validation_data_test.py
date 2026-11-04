# expect: LK005
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

data = pd.read_parquet("signals.parquet").drop_duplicates()
X, y = data.drop(columns="label").values, data["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
net = keras.Sequential([keras.layers.Dense(64, activation="relu"), keras.layers.Dense(1, activation="sigmoid")])
net.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
net.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test),
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
print(net.evaluate(X_test, y_test))
