# expect: LK015
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X = np.load("embeddings.npy")
y = np.load("labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300).fit(X_train, y_train)
print(mlp.score(X_test, y_test))
