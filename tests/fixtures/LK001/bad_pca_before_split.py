import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
df = pd.read_parquet("data.parquet").drop_duplicates()
X = df.drop("label", axis=1)
y = df["label"]
X_pca = PCA(n_components=10).fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_pca, y, test_size=0.25)
