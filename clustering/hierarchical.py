from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MyAgglomerative:
    def __init__(self, n_clusters: int = 3, metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        possible_metrics = ['euclidean', 'chebyshev', 'manhattan', 'cosine']
        if metric not in possible_metrics:
            raise ValueError(f'Unknown ``metric``={metric} parameter.'
                             f' Should be one of: [{", ".join(possible_metrics)}]')
        self.metric = metric

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

    def fit_predict(self, X: pd.DataFrame):
        last_cluster_id = 0
        det_clusters = pd.Series(index=X.index)
        centroids = [[row] for _, row in X.iterrows()]
        while len(centroids) > self.n_clusters:
            best_i = best_j = -1
            min_dst = float('inf')
            for i in range(len(centroids) - 1):
                item1 = np.mean(centroids[i], axis=0)
                for j in range(i + 1, len(centroids)):
                    item2 = np.mean(centroids[j], axis=0)
                    dst = self._get_distance(item1, item2)
                    if dst < min_dst:
                        min_dst, best_i, best_j = dst, i, j
            cluster_id = det_clusters[centroids[best_i][0].name]
            if pd.isna(cluster_id):
                det_clusters[centroids[best_i][0].name] = cluster_id = last_cluster_id
                last_cluster_id += 1
            for row in centroids[best_j]:
                det_clusters[row.name] = cluster_id
            centroids[best_i].extend(centroids.pop(best_j))
        return det_clusters

    def __str__(self):
        return (f'{self.__class__.__name__} class: n_clusters={self.n_clusters}')
