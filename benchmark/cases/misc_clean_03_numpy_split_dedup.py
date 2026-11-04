# expect: clean
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X = np.load("embeddings.npy")
y = np.load("labels.npy")
X, idx = np.unique(X, axis=0, return_index=True)
y = y[idx]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
print(MLPClassifier(max_iter=300).fit(X_train, y_train).score(X_test, y_test))
