from abc import ABC
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class MyKNN(ABC):
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        if metric not in ['euclidean', 'manhattan', 'chebyshev', 'cosine']:
            raise ValueError(f'Unknown ``metric``={metric} parameter')
        self.metric = metric
        if weight not in ['uniform', 'rank', 'distance']:
            raise ValueError(f'Unknown ``weight``={weight} parameter')
        self.weight = weight
        self.train_size: Optional[Tuple[int, int]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def _get_distance(self, x1: pd.Series, x2: pd.Series) -> float:
        if self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.metric == 'manhattan':
            return np.linalg.norm(x1 - x2, ord=1)
        elif self.metric == 'chebyshev':
            return max(np.abs(x1 - x2))
        elif self.metric == 'cosine':
            return 1 - x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return np.inf

    def _get_k_neighbours(self, row: pd.Series) -> pd.DataFrame:
        distance_df = self.X_train.apply(
            self._get_distance, axis=1, x2=row, result_type='reduce').to_frame('distance')
        distance_label_df = distance_df.assign(label=self.y_train)
        nearest_neighbours = distance_label_df.nsmallest(self.k, 'distance')
        return nearest_neighbours

    def _get_weights(self, neighbours_df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def _get_label(self, row: pd.Series) -> Union[int, float]:
        raise NotImplementedError()

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        return X_test.apply(self._get_label, axis=1)

    def __str__(self):
        return (f'{self.__class__.__name__} class: k={self.k}, metric={self.metric},'
                f'weight={self.weight}, train_size={self.train_size}')
