from typing import Optional

import pandas as pd

from .decision_tree_base import MyTree


class MyTreeClf(MyTree):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 criterion: str = 'entropy', bins: Optional[int] = None):
        possible_criterions = ['entropy', 'gini']
        if criterion not in possible_criterions:
            raise ValueError('Unknown gain function option in ``criterion`` parameter.'
                             f' Should be one of: [{", ".join(possible_criterions)}]')
        super().__init__(max_depth, min_samples_split, max_leafs, criterion, bins)

    def _calculate_leaf_value(self, y: pd.Series) -> float:
        return y.mean()  # mode().iloc[-1]

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return super().predict(X)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return (self.predict_proba(X) > 0.5).astype(int)
