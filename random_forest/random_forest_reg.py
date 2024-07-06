import inspect

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from decision_tree.decision_tree_reg import MyTreeReg

from .random_forest_base import MyForest


class MyForestReg(MyForest):
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5,
                 random_state: int = 42,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16,
                 oob_score: Optional[str] = 'mae'):
        possible_oob_score_metrics = ['mae', 'rmse', 'mse', 'mape', 'r2']
        if oob_score not in possible_oob_score_metrics:
            raise ValueError(f'Unknown ``oob_score``={oob_score} parameter.'
                             f' Should be one of: [{", ".join(possible_oob_score_metrics)}]')
        super().__init__(n_estimators, max_features, max_samples, random_state,
                         max_depth, min_samples_split, max_leafs, bins, oob_score)
        self._TreeClass = MyTreeReg
        for param_name, param in inspect.signature(self._TreeClass.__init__).parameters.items():
            if param_name != 'self':
                default = None
                if param.default is not param.empty:
                    default = param.default
                self._tree_params[param_name] = default

    def _get_oob_score(self, oob_predictions: Dict[str, List[float]], y: pd.Series) -> float:
        y_hat = pd.Series(np.empty(len(oob_predictions), float), index=list(oob_predictions.keys()))
        for idx, predictions in oob_predictions.items():
            y_hat[idx] = np.mean(predictions)

        if self.oob_metric_ == 'mae':
            return np.mean(abs(y_hat - y))
        elif self.oob_metric_ == 'rmse':
            return np.sqrt(np.mean((y_hat - y) ** 2))
        elif self.oob_metric_ == 'mse':
            return np.mean((y_hat - y) ** 2)
        elif self.oob_metric_ == 'mape':
            return 100 * np.mean(abs((y - y_hat) / y))
        elif self.oob_metric_ == 'r2':
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)
