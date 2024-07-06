from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .linear_base import MyLinear


class MyLineReg(MyLinear):
    def __init__(self, n_iter: int = 100, learning_rate: Union[float, Callable] = 0.1,
                 weights: Optional[Union[np.ndarray, pd.Series]] = None,
                 metric: Optional[str] = 'mse',
                 reg: Optional[str] = None, l1_coef: float = 0, l2_coef: float = 0,
                 sgd_sample: Optional[Union[int, float]] = None, random_state: int = 42):
        possible_metrics = ['mae', 'rmse', 'mse', 'mape', 'r2']
        if metric not in possible_metrics:
            raise ValueError(f'Unknown ``metric``={metric} parameter.'
                             f' Should be one of: [{", ".join(possible_metrics)}]')
        super().__init__(n_iter, learning_rate, weights, metric, reg, l1_coef, l2_coef, sgd_sample, random_state)

    def _linear_combination(self, X: pd.DataFrame) -> pd.Series:
        return X @ self.W

    def _count_loss(self, prediction: pd.Series, target: pd.Series) -> float:
        return (
            np.mean((prediction - target) ** 2)
            + self.l1_coef * abs(self.W).sum() + self.l2_coef * (self.W ** 2).sum()
        )

    def _count_grad(self, prediction: pd.Series, target: pd.Series, X: pd.DataFrame) -> pd.Series:
        return 2 * ((prediction - target) @ X) / len(X) + self.l1_coef * np.sign(self.W) + self.l2_coef * 2 * self.W

    def get_best_score(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> float:
        if X is None and y is None:
            return self._last_score
        y_hat = self.predict(X)
        if self.metric == 'mae':
            return np.mean(abs(y_hat - y))
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y_hat - y) ** 2))
        elif self.metric == 'mse':
            return np.mean((y_hat - y) ** 2)
        elif self.metric == 'mape':
            return 100 * np.mean(abs((y - y_hat) / y))
        elif self.metric == 'r2':
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
