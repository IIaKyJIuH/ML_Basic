from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MyPCA:
    def __init__(self, n_components: int = 3):
        self.n_components = n_components

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        col_mean = np.mean(X, axis=0)
        normalised = X.copy() - col_mean
        eigenvalues, eigenvectors = np.linalg.eigh(normalised.cov())
        sorted_indices = np.argsort(eigenvalues)[::-1]
        s_eigenvectors = eigenvectors.T[sorted_indices]
        main_components = s_eigenvectors[:self.n_components]

        return normalised @ main_components.T

    def __str__(self):
        return (f'{self.__class__.__name__} class: n_components={self.n_components}')
