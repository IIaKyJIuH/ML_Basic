from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class MyKMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 10, n_init: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_: List[np.ndarray] = []
        self.inertia_: float = float('inf')

    def _randomise_centroids(self, X: pd.DataFrame):
        return [
            np.array([
                np.random.uniform(col.min(), col.max())
                for _, col in X.items()
            ])
            for _ in range(self.n_clusters)
        ]

    def count_wcss(self, det_clusters: List[List[pd.Series]]):
        return sum(
            sum(np.linalg.norm(row - self.cluster_centers_[clst_id]) ** 2 for row in cluster_points)
            for clst_id, cluster_points in enumerate(det_clusters)
        )

    def fit(self, X: pd.DataFrame):
        np.random.seed(seed=self.random_state)
        best_centers = []
        best_inertia = float('inf')
        for _ in range(self.n_init):
            self.cluster_centers_ = self._randomise_centroids(X)
            cur_iter = 0
            prev_centers = [np.full(len(X.columns), 0.) for _ in range(self.n_clusters)]
            while cur_iter < self.max_iter and not np.allclose(self.cluster_centers_, prev_centers):
                det_clusters = [[] for _ in range(self.n_clusters)]
                for _, row in X.iterrows():
                    min_dst, min_clst = float('inf'), -1
                    for clst_id, centroid in enumerate(self.cluster_centers_):
                        dst = np.linalg.norm(row - centroid)
                        if dst < min_dst:
                            min_dst, min_clst = dst, clst_id
                    det_clusters[min_clst].append(row)

                prev_centers = self.cluster_centers_

                self.cluster_centers_ = [
                    np.mean(centroid_points, axis=0) if centroid_points else self.cluster_centers_[clst_id]
                    for clst_id, centroid_points in enumerate(det_clusters)
                ]

                cur_iter += 1
            inertia = self.count_wcss(det_clusters)
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = self.cluster_centers_
        self.inertia_ = best_inertia
        self.cluster_centers_ = best_centers

    def _determine_cluster(self, row: pd.Series) -> int:
        min_dst, min_clst = float('inf'), -1
        for clst_id, centroid in enumerate(self.cluster_centers_):
            dst = np.linalg.norm(row - centroid)
            if dst < min_dst:
                min_dst, min_clst = dst, clst_id
        return min_clst

    def predict(self, X: pd.DataFrame):
        return X.apply(self._determine_cluster, axis=1)

    def __str__(self):
        return (f'{self.__class__.__name__} class: n_clusters={self.n_clusters}, max_iter={self.max_iter},'
                f' n_init={self.n_init}, random_state={self.random_state}')
