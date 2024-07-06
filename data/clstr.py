import pandas as pd

from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, centers=5, n_features=5, cluster_std=2.5, random_state=42)
X = pd.DataFrame(X)
X.columns = [f'col_{col}' for col in X.columns]
