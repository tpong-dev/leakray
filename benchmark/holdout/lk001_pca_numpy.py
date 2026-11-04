# expect: LK001
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

raw = pd.read_csv("signals.csv").drop_duplicates()
target = raw.pop("label").to_numpy()
arr = raw.to_numpy(dtype="float32")

reduced = PCA(n_components=10, random_state=0).fit_transform(arr)
a_tr, a_te, t_tr, t_te = train_test_split(reduced, target, test_size=0.25, random_state=4)
clf = SVC(kernel="rbf").fit(a_tr, t_tr)
print(clf.score(a_te, t_te))
