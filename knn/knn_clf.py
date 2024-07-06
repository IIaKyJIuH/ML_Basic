from typing import Union

import numpy as np
import pandas as pd

from .knn_base import MyKNN


class MyKNNClf(MyKNN):
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        super().__init__(k, metric, weight)

    def _get_weights(self, neighbours_df: pd.DataFrame) -> pd.Series:
        if self.weight == 'rank':
            sorted_df = neighbours_df.sort_values('distance').reset_index()
            sorted_df.index += 1
            Rij_sum = sorted_df.groupby('label').agg(lambda distance: np.sum(1 / distance.index))['distance']
            Ri_sum = np.sum(1 / sorted_df.index)
            return Rij_sum / Ri_sum
        elif self.weight == 'distance':
            Dij_sum = neighbours_df.groupby('label').agg(lambda distance: np.sum(1 / distance))['distance']
            Di_sum = np.sum(1 / neighbours_df['distance'])
            return Dij_sum / Di_sum

    def _get_label(self, row: pd.Series) -> Union[int, float]:
        neighbours_df = self._get_k_neighbours(row)
        labels = neighbours_df['label']
        if self.weight == 'uniform':
            return labels.mode().iloc[-1]
        elif self.weight in ['rank', 'distance']:
            return self._get_weights(neighbours_df).idxmax()

    def _get_proba(self, row: pd.Series, *, target_label: int = 1) -> float:
        neighbours_df = self._get_k_neighbours(row)
        labels = neighbours_df['label']
        if self.weight == 'uniform':
            return len(labels[labels == target_label]) / len(labels)
        elif self.weight in ['rank', 'distance']:
            weights = self._get_weights(neighbours_df)
            if target_label not in weights:
                return 0
            return weights[target_label]

    def predict_proba(self, X_test: pd.DataFrame, *, target_label: int = 1) -> pd.Series:
        return X_test.apply(self._get_proba, target_label=target_label, axis=1)
