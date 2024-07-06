from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .linear_base import MyLinear


class MyLogReg(MyLinear):
    def __init__(self, n_iter: int = 10, learning_rate: Union[float, Callable] = 0.1,
                 weights: Optional[Union[np.ndarray, pd.Series]] = None,
                 metric: Optional[str] = 'accuracy',
                 reg: Optional[str] = None, l1_coef: float = 0, l2_coef: float = 0,
                 sgd_sample: Optional[Union[int, float]] = None, random_state: int = 42):
        possible_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        if metric not in possible_metrics:
            raise ValueError(f'Unknown ``metric``={metric} parameter.'
                             f' Should be one of: [{", ".join(possible_metrics)}]')
        super().__init__(n_iter, learning_rate, weights, metric, reg, l1_coef, l2_coef, sgd_sample, random_state)

    def _linear_combination(self, X: pd.DataFrame) -> pd.Series:
        return 1 / (1 + np.exp(-X @ self.W))

    def _count_loss(self, prediction: pd.Series, target: pd.Series) -> float:
        return (
            -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
            + self.l1_coef * abs(self.W).sum() + self.l2_coef * (self.W ** 2).sum()
        )

    def _count_grad(self, prediction: pd.Series, target: pd.Series, X: pd.DataFrame) -> pd.Series:
        return ((prediction - target) @ X) / len(X) + self.l1_coef * np.sign(self.W) + self.l2_coef * 2 * self.W

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return super().predict(X)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)

    def get_best_score(self, X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None) -> float:
        if X is None and y is None:
            return self._last_score
        y_hat = self.predict(X)
        if self.metric == 'accuracy':
            return (y_hat == y).mean()
        elif self.metric in ['precision', 'recall', 'f1']:
            TP = np.sum(y & y_hat)
            precision = TP / np.sum(y_hat)
            recall = TP / np.sum(y)
            if self.metric == 'precision':
                return precision
            elif self.metric == 'recall':
                return recall
            return 2 * precision * recall / (precision + recall)
        elif self.metric == 'roc_auc':
            y_hat_probs = self.predict_proba(X)
            PN = y.sum() * (len(y) - y.sum())
            y_hat_probs = round(y_hat_probs, 10)  # for the sake of reproducibility
            table = np.array(
                list(zip(np.around(y_hat_probs, 10), y)),
                dtype=[('probs', float), ('labels', int)]
            )
            sorted_probs_arg = np.argsort(table, order=['probs', 'labels'])[::-1]
            sorted_labels = y.iloc[sorted_probs_arg]
            sorted_probs = y_hat_probs.iloc[sorted_probs_arg]
            positive_idxs = []
            result_sum = 0.
            for idx, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
                if label == 0:
                    pos_probs_before = sorted_probs.iloc[positive_idxs]
                    above_pos_probs_cnt = np.sum(pos_probs_before > prob)
                    equal_pos_probs_cnt = len(pos_probs_before) - above_pos_probs_cnt
                    result_sum += equal_pos_probs_cnt / 2 + above_pos_probs_cnt
                else:
                    positive_idxs.append(idx)
            return result_sum / PN
