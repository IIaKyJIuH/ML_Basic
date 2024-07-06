import pandas as pd

from sklearn.datasets import make_regression

reg_X, reg_y = make_regression(n_samples=150, n_features=14, n_informative=10, noise=15, random_state=42)
reg_X = pd.DataFrame(reg_X)
reg_y = pd.Series(reg_y)
reg_X.columns = [f'col_{col}' for col in reg_X.columns]

# Diabetes
from sklearn.datasets import load_diabetes

diab_data = load_diabetes(as_frame=True)
diab_X, diab_y = diab_data['data'], diab_data['target']
