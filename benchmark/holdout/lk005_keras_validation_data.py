# expect: LK005
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras

frame = pd.read_csv("digits_flat.csv").drop_duplicates()
y = frame.pop("digit").to_numpy()
x = frame.to_numpy(dtype="float32") / 255.0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

net = keras.Sequential([
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax"),
])
net.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
net.fit(
    x_train, y_train, epochs=50, batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
)
print(net.evaluate(x_test, y_test))
