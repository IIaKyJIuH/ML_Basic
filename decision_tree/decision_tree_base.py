from abc import ABC
from collections import defaultdict
from typing import Dict, Hashable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class Node():
    def __init__(self, col_name: Optional[Union[int, str]] = None, split_value: Optional[int] = None,
                 ig: Optional[float] = None, left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[Union[int, float]] = None):
        # for a decision node
        self.col_name = col_name
        self.split_value = split_value
        self.ig = ig
        self.left = left
        self.right = right

        # for a leaf node
        self.value = value


class MyTree(ABC):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 criterion: str = 'entropy', bins: Optional[int] = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.criterion = criterion
        self.bins = bins

        self.leafs_cnt: int = 0
        self.root: Optional[Node] = None
        self._thresholds: Dict[Hashable, np.ndarray] = {}
        self.fi: Dict[str, Union[int, float]] = defaultdict(int)

    def _compute_stats(self, y: pd.Series) -> float:
        if self.criterion == 'variance':
            return y.var()
        elif self.criterion == 'mse':
            return ((y - y.mean()) ** 2).mean()
        elif self.criterion == 'gini':
            func = lambda p_cls: p_cls ** 2
            return_func = lambda result: 1 - result
        elif self.criterion == 'entropy':
            func = lambda p_cls: p_cls * np.log2(p_cls) if p_cls != 0 else 0
            return_func = lambda result: -result
        statistics = (y.value_counts() / len(y)).apply(func).sum()
        return return_func(statistics)

    def _information_gain(self, parent: pd.Series, left_child: pd.Series, right_child: pd.Series):
        ''' function to compute information gain '''

        l_weight = len(left_child) / len(parent)
        r_weight = len(right_child) / len(parent)
        gain = (
            self._compute_stats(parent) -
            (l_weight * self._compute_stats(left_child) + r_weight * self._compute_stats(right_child))
        )
        return gain

    def _get_thresholds(self, data: pd.Series) -> np.ndarray:
        if self._thresholds:
            return self._thresholds[data.name]
        unique_srtd = np.unique(data)
        return (unique_srtd[:-1] + unique_srtd[1:]) / 2

    def _build_histogram(self, X: pd.DataFrame):
        if self.bins is not None:
            for column_name, column in X.items():
                unique_srtd = np.unique(column)
                thresholds = (unique_srtd[:-1] + unique_srtd[1:]) / 2
                if len(thresholds) > self.bins - 1:
                    thresholds = np.histogram(column, self.bins)[1][1:-1]
                self._thresholds[column_name] = thresholds

    def _get_best_split(self, X: pd.DataFrame, y: pd.Series) -> dict:
        ''' function to find the best split for current parent (y) and data (X)'''

        # dictionary to store the best split
        best_split = {}
        max_info_gain = 0
        # loop over all features
        for column_name, column in X.items():
            possible_thresholds = self._get_thresholds(column)
            for threshold in possible_thresholds:
                # get current split
                is_right_tree = column > threshold
                left_subtree, right_subtree = X[~is_right_tree], X[is_right_tree]
                # check if childs are not null
                if not (left_subtree.empty or right_subtree.empty):
                    left_y, right_y = [y.loc[subtree.index] for subtree in [left_subtree, right_subtree]]
                    # compute information gain
                    curr_info_gain = self._information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split['col_name'] = column_name
                        best_split['split_value'] = threshold
                        best_split['ig'] = curr_info_gain
                        best_split['left_dataset'] = (left_subtree, left_y)
                        best_split['right_dataset'] = (right_subtree, right_y)
                        max_info_gain = curr_info_gain
        return best_split

    def _calculate_leaf_value(self, y: pd.Series) -> Union[int, float]:
        return y.mean()

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, cur_depth: int = 0,
                    leaves_available: int = 0) -> Tuple[Optional[Node], int]:
        # split until stopping conditions are met
        if ((len(pd.unique(y)) > 1 and len(X) >= self.min_samples_split and cur_depth < self.max_depth
                and self.leafs_cnt + 2 < self.max_leafs) or cur_depth == 0):
            # find the best split
            best_split = self._get_best_split(X, y)
            # check if information gain is positive
            if best_split:
                left_dataset, right_dataset = best_split['left_dataset'], best_split['right_dataset']
                # recur left
                left_subtree, l_cnt = self._build_tree(*left_dataset, cur_depth=cur_depth + 1,
                                                       leaves_available=leaves_available)
                # recur right
                right_subtree, r_cnt = self._build_tree(*right_dataset, cur_depth=cur_depth + 1,
                                                        leaves_available=leaves_available - l_cnt)
                # return decision node
                if left_subtree is not None and right_subtree is not None:
                    self.fi[best_split['col_name']] += len(X) * best_split['ig']
                    return Node(best_split['col_name'], best_split['split_value'], best_split['ig'],
                                left_subtree, right_subtree), l_cnt + r_cnt
                else:
                    self.leafs_cnt -= l_cnt + r_cnt
        if self.leafs_cnt < self.max_leafs or len(pd.unique(y)) == 1 or len(X) == 1:
            self.leafs_cnt += 1
            # compute leaf node
            leaf_value = self._calculate_leaf_value(y)
            # return leaf node
            return Node(value=leaf_value), 1
        return None, 0

    def fit(self, X: pd.DataFrame, y: pd.Series, fi_normalisation: Optional[int] = None):
        ''' function to train the tree '''
        self._build_histogram(X)
        self.root, _ = self._build_tree(X, y, 0, self.max_leafs)
        for column in X.columns:
            if self.fi[column]:
                self.fi[column] *= 1 / (fi_normalisation or len(X))

    def _make_prediction(self, row: pd.Series, tree: Optional[Node] = None) -> float:
        ''' function to predict a single data point '''
        if tree is None:
            raise ValueError('The tree was built incorrectly: the tree descent has led to None')

        if tree.value is not None:
            return tree.value
        if tree.col_name is None:
            raise ValueError('The tree was built incorrectly: you are trying to descent from a leaf of the tree')
        feature_val = row[tree.col_name]
        if feature_val > tree.split_value:
            return self._make_prediction(row, tree.right)
        return self._make_prediction(row, tree.left)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._make_prediction, tree=self.root, axis=1)

    def print_tree(self, tree: Optional[Node] = None, indent: str = '', print_with_nums: bool = False):
        if tree is None:
            if indent:
                return 0
            tree = self.root
            if print_with_nums:
                indent = '1'
            if tree is None:
                raise ValueError('The tree is not built yet')

        nodes_sm = 0
        if tree.value is not None:
            nodes_sm = tree.value
        else:
            print(f'{indent} - {tree.col_name} > {tree.split_value} ? info_gain={tree.ig}')
            indent += '.' if print_with_nums else ' '
            is_next_left_subtree = tree.left.split_value is not None
            is_next_right_subtree = tree.right.split_value is not None
            if is_next_left_subtree:
                nodes_sm += self.print_tree(tree.left, indent + '1' if print_with_nums else '', print_with_nums)
            if is_next_right_subtree:
                nodes_sm += self.print_tree(tree.right, indent + '2' if print_with_nums else '', print_with_nums)
            if not is_next_left_subtree:
                node_val = self.print_tree(tree.left, indent)
                print(f'{indent}left - {node_val}')
                nodes_sm += node_val
            if not is_next_right_subtree:
                node_val = self.print_tree(tree.right, indent)
                print(f'{indent}right - {node_val}')
                nodes_sm += node_val
        return nodes_sm

    def __str__(self):
        return (f'{self.__class__.__name__} class: max_depth={self.max_depth},'
                f' min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs},'
                f' criterion={self.criterion}, bins={self.bins}')
