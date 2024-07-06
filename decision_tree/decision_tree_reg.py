from typing import Optional

from .decision_tree_base import MyTree


class MyTreeReg(MyTree):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 criterion: str = 'mse', bins: Optional[int] = None):
        possible_criterions = ['mse', 'variance']
        if criterion not in possible_criterions:
            raise ValueError('Unknown gain function option in ``criterion`` parameter.'
                             f' Should be one of: [{", ".join(possible_criterions)}]')
        super().__init__(max_depth, min_samples_split, max_leafs, criterion, bins)
