from pathlib import Path

import pandas as pd

from sklearn.datasets import make_classification

clf_X, clf_y = make_classification(n_samples=1000, n_features=14, n_informative=10, random_state=42)
clf_X = pd.DataFrame(clf_X)
clf_y = pd.Series(clf_y)
clf_X.columns = [f'col_{col}' for col in clf_X.columns]

# Banknote Authentication
bank_df = pd.read_csv(Path(Path(__file__).resolve().parent, 'data_banknote_authentication.txt'), header=None)
bank_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'target']
bank_X, bank_y = bank_df.iloc[:, :4], bank_df['target']
