import random

from abc import ABC
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd


class MyLinear(ABC):
    def __init__(self, n_iter: int = 50, learning_rate: Union[float, Callable] = 0.1,
                 weights: Optional[Union[np.ndarray, pd.Series]] = None,
                 metric: Optional[str] = None,
                 reg: Optional[str] = None, l1_coef: float = .0, l2_coef: float = .0,
                 sgd_sample: Optional[Union[int, float]] = None, random_state: int = 42):
        self.n_iter = n_iter
        if not callable(learning_rate):
            self.learning_rate = lambda *args: learning_rate
        else:
            self.learning_rate = learning_rate
        self.W = weights
        self.metric = metric
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        if reg is None:
            self.l1_coef = self.l2_coef = 0
        elif reg == 'l1':
            self.l2_coef = 0
        elif reg == 'l2':
            self.l1_coef = 0
        self.sgd_sample = sgd_sample or 1.
        self.random_state = random_state

        self._last_score = None

    def _insert_ones_column(self, X: pd.DataFrame) -> pd.DataFrame:
        if 'x0' not in X.columns:
            X = X.copy()
            X.insert(0, 'x0', np.ones(len(X)))
        return X

    def _linear_combination(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def _count_loss(self, prediction: pd.Series, target: pd.Series) -> float:
        raise NotImplementedError()

    def _count_grad(self, prediction: pd.Series, target: pd.Series, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: Union[int, bool] = False):
        random.seed(self.random_state)
        _X = self._insert_ones_column(X)
        self.W = np.ones(len(_X.columns))
        if isinstance(self.sgd_sample, float):
            self.sgd_sample = int(len(_X) * self.sgd_sample)
        for i in range(1, self.n_iter + 1):
            if verbose:
                z = self._linear_combination(_X)
                loss = self._count_loss(z, y)
                metric_score = ''
                if self.metric is not None:
                    self._last_score = self.get_best_score(_X, y)
                    metric_score = f' | {self.metric}: {self._last_score}'
                if i % verbose == 0 or i == 1:
                    print(f'{"start" if i == 1 else i} | loss: {loss}{metric_score}')
            sample_rows_idx = random.sample(range(_X.shape[0]), self.sgd_sample)
            sub_X = _X.iloc[sample_rows_idx]
            sub_y = y.iloc[sample_rows_idx]
            y_hat = self._linear_combination(sub_X)
            grad = self._count_grad(y_hat, sub_y, sub_X)
            step = -grad
            self.W += self.learning_rate(i) * step
        self._last_score = self.get_best_score(_X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self._linear_combination(self._insert_ones_column(X))

    def get_best_score(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None):
        raise NotImplementedError()

    def get_coef(self):
        return self.W[1:]

    def __str__(self):
        return (f'{self.__class__.__name__} class: n_iter={self.n_iter}, learning_rate={self.learning_rate()},'
                f'metric={self.metric}')
