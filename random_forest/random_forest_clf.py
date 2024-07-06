import inspect

from functools import reduce
from typing import Dict, Hashable, List, Optional

import numpy as np
import pandas as pd

from decision_tree.decision_tree_clf import MyTreeClf

from .random_forest_base import MyForest


class MyForestClf(MyForest):
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5,
                 random_state: int = 42,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16,
                 oob_score: Optional[str] = 'accuracy', criterion: str = 'entropy'):
        possible_oob_score_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        if oob_score not in possible_oob_score_metrics:
            raise ValueError(f'Unknown ``oob_score``={oob_score} parameter.'
                             f' Should be one of: [{", ".join(possible_oob_score_metrics)}]')
        super().__init__(n_estimators, max_features, max_samples, random_state,
                         max_depth, min_samples_split, max_leafs, bins, oob_score)
        self._TreeClass = MyTreeClf
        for param_name, param in inspect.signature(self._TreeClass.__init__).parameters.items():
            if param_name != 'self':
                default = None
                if param.default is not param.empty:
                    default = param.default
                self._tree_params[param_name] = default
        if criterion not in ['entropy', 'gini']:
            raise ValueError(f'Unknown ``criterion``={criterion} parameter')
        self.criterion = criterion
        self._estimators: List[MyTreeClf] = []

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        results = []
        for estimator in self._estimators:
            results.append(estimator.predict_proba(X))
        return reduce(lambda x, y: x + y, results) / len(results)

    def predict(self, X: pd.DataFrame, type: str = 'mean'):
        if type == 'mean':
            return (self.predict_proba(X) > 0.5).astype(int)
        elif type == 'vote':
            results = []
            for estimator in self._estimators:
                results.append(estimator.predict(X))
            return (reduce(lambda x, y: x + y, results) / len(results) >= 0.5).astype(int)

    def _predict_oob(self, estimator: MyTreeClf, oob_dict: Dict[Hashable, List[float]], data: pd.DataFrame):
        for idx, prediction in estimator.predict_proba(data).items():
            oob_dict[idx].append(prediction)

    def _get_oob_score(self, oob_predictions: Dict[str, List[float]], y: pd.Series) -> float:
        y_hat_probs = pd.Series(np.empty(len(oob_predictions), float), index=list(oob_predictions.keys()))
        for idx, predictions in oob_predictions.items():
            y_hat_probs[idx] = np.mean(predictions)
        y_hat = (y_hat_probs > 0.5).astype(int)

        if self.oob_metric_ == 'accuracy':
            return (y_hat == y).mean()
        elif self.oob_metric_ in ['precision', 'recall', 'f1']:
            TP = np.sum(y & y_hat)
            precision = TP / np.sum(y_hat)
            recall = TP / np.sum(y)
            if self.oob_metric_ == 'precision':
                return precision
            elif self.oob_metric_ == 'recall':
                return recall
            return 2 * precision * recall / (precision + recall)
        elif self.oob_metric_ == 'roc_auc':
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

    def __str__(self):
        return (f'{self.__class__.__name__} class: n_estimators={self.n_estimators}, max_features={self.max_features},'
                f' max_samples={self.max_samples}, max_depth={self.max_depth},'
                f' min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins},'
                f' criterion={self.criterion}, random_state={self.random_state}')
