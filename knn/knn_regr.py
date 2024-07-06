from typing import Union

import pandas as pd

from .knn_base import MyKNN


class MyKNNReg(MyKNN):
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        super().__init__(k, metric, weight)

    def _get_weights(self, neighbours_df: pd.DataFrame) -> pd.Series:
        if self.weight == 'rank':
            sorted_df = neighbours_df.sort_values('distance')
            original_index = sorted_df.index
            sorted_df.reset_index(inplace=True)
            sorted_df.index += 1
            R = 1 / sorted_df.index.to_series()
            W = R / R.sum()
            W.index = original_index
        elif self.weight == 'distance':
            D = 1 / neighbours_df['distance']
            W = D / D.sum()
        return W

    def _get_label(self, row: pd.Series) -> Union[int, float]:
        neighbours_df = self._get_k_neighbours(row)
        labels = neighbours_df['label']
        if self.weight == 'uniform':
            return labels.mean()
        elif self.weight in ['rank', 'distance']:
            return labels @ self._get_weights(neighbours_df)
