from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MyDBSCAN:
    def __init__(self, eps: int = 3, min_samples: int = 3, metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        possible_metrics = ['euclidean', 'chebyshev', 'manhattan', 'cosine']
        if metric not in possible_metrics:
            raise ValueError(f'Unknown ``metric``={metric} parameter.'
                             f' Should be one of: [{", ".join(possible_metrics)}]')
        self.metric = metric

        self._last_cluster = 0

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

    def _scan(self, X: pd.DataFrame, parent_neighbours: Union[list, range], visited: list, outliers: set):
        for i in parent_neighbours:
            if visited[i] == -1:
                if i in outliers and not isinstance(parent_neighbours, range):
                    visited[i] = self._last_cluster
                    outliers.remove(i)
                    continue
                item1 = X.iloc[i]
                neighbours = []
                for j in range(len(X)):
                    if i != j:
                        item2 = X.iloc[j]
                        dst = self._get_distance(item1, item2)
                        if dst < self.eps:
                            neighbours.append(j)
                if len(neighbours) < self.min_samples:
                    if not isinstance(parent_neighbours, range):
                        visited[i] = self._last_cluster
                    else:
                        outliers.add(i)
                else:
                    visited[i] = self._last_cluster
                    self._scan(X, neighbours, visited, outliers)
                    if isinstance(parent_neighbours, range):
                        self._last_cluster += 1

    def fit_predict(self, X: pd.DataFrame):
        visited = [-1 for _ in range(len(X))]
        outliers = set()
        self._scan(X, range(len(X)), visited, outliers)
        for i in outliers:
            visited[i] = self._last_cluster
        return visited

    def __str__(self):
        return (f'{self.__class__.__name__} class: eps={self.eps}, min_samples={self.min_samples}')
