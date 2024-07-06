import inspect
import random

from abc import ABC
from collections import defaultdict
from functools import partial, reduce
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple, Type, Union

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
        if criterion not in ['entropy', 'gini', 'mse', 'variance']:
            raise ValueError('Unknown gain function option in ``criterion`` parameter')
        self.criterion = criterion
        self.bins = bins

        self.leafs_cnt: int = 0
        self.root: Optional[Node] = None
        self._thresholds: Dict[str, np.ndarray] = {}
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
                is_left_tree = column <= threshold
                left_split, right_split = X[is_left_tree], X[~is_left_tree]
                # check if childs are not null
                if not (left_split.empty or right_split.empty):
                    left_y, right_y = [y.loc[split.index] for split in [left_split, right_split]]
                    # compute information gain
                    curr_info_gain = self._information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split['col_name'] = column_name
                        best_split['split_value'] = threshold
                        best_split['ig'] = curr_info_gain
                        best_split['left_dataset'] = (left_split, left_y)
                        best_split['right_dataset'] = (right_split, right_y)
                        max_info_gain = curr_info_gain
        return best_split

    def _calculate_leaf_value(self, y: pd.Series) -> Union[int, float]:
        raise NotImplementedError()

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, cur_depth: int = 0,
                    leaves_available: int = 0) -> Tuple[Optional[Node], int]:
        # split until stopping conditions are met
        if ((len(X) >= self.min_samples_split and cur_depth < self.max_depth and leaves_available >= 2) or
                cur_depth == 0):
            # find the best split
            best_split = self._get_best_split(X, y)
            # check if information gain is positive
            if best_split:
                left_dataset, right_dataset = best_split['left_dataset'], best_split['right_dataset']
                if len(left_dataset[1]) and len(right_dataset[1]):
                    # recur left
                    left_subtree, l_cnt = self._build_tree(*left_dataset, cur_depth + 1, leaves_available - 1)
                    # recur right
                    right_subtree, r_cnt = self._build_tree(*right_dataset, cur_depth + 1, leaves_available - l_cnt)
                    # return decision node
                    if left_subtree is not None and right_subtree is not None:
                        self.fi[best_split['col_name']] += len(X) * best_split['ig']
                        return Node(best_split['col_name'], best_split['split_value'], best_split['ig'],
                                    left_subtree, right_subtree), l_cnt + r_cnt
        if leaves_available >= 0:
            # create a leaf
            self.leafs_cnt += 1
            # compute leaf node
            leaf_value = self._calculate_leaf_value(y)
            # return leaf node
            return Node(value=leaf_value), 1
        return None, 0

    def fit(self, X: pd.DataFrame, y: pd.Series, fi_normalisation: Optional[int] = None):
        ''' function to train the tree '''
        if self.bins is not None:
            self._build_histogram(X)
        self.root, _ = self._build_tree(X, y, 0, self.max_leafs)
        for column in X.columns:
            if self.fi[column]:
                self.fi[column] *= 1 / (fi_normalisation or len(X))

    def _make_prediction(self, row: pd.Series, tree: Node) -> float:
        ''' function to predict a single data point '''

        if tree.value is not None:
            return tree.value
        feature_val = row[tree.col_name]
        if feature_val <= tree.split_value:
            return self._make_prediction(row, tree.left)
        return self._make_prediction(row, tree.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def print_tree(self, tree: Optional[Node] = None, indent: str = ''):
        if tree is None:
            return

        nodes_sm = 0
        if tree.value is not None:
            nodes_sm = tree.value
            print(nodes_sm)
        else:
            print(f'{indent}{tree.col_name} <= {tree.split_value} ? info_gain={tree.ig}')
            indent += ' '
            is_next_left_leaf = tree.left is not None and tree.left.value is not None
            if is_next_left_leaf:
                print(f'{indent}{"leaf_l = "}', end='')
            nodes_sm += self.print_tree(tree.left, indent)
            is_next_right_leaf = tree.right is not None and tree.right.value is not None
            if is_next_right_leaf:
                print(f'{indent}{"leaf_r = "}', end='')
            nodes_sm += self.print_tree(tree.right, indent)
        return nodes_sm

    def __str__(self):
        return (f'{self.__class__.__name__} class: max_depth={self.max_depth},'
                f' min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs},'
                f' criterion={self.criterion}, bins={self.bins}')


class MyTreeReg(MyTree):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 criterion: str = 'mse', bins: Optional[int] = None):
        super().__init__(max_depth, min_samples_split, max_leafs, criterion, bins)

    def _calculate_leaf_value(self, y: pd.Series) -> float:
        return y.mean()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._make_prediction, tree=self.root, axis=1)


class MyForest(ABC):
    def __init__(self, n_estimators: int = 10, max_features: float = .5, max_samples: float = .5,
                 random_state: int = 42,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16,
                 oob_score: Optional[str] = None, use_multiprocessing: bool = False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        if oob_score not in ['mae', 'rmse', 'mse', 'mape', 'r2']:
            raise ValueError(f'Unknown ``oob_score``={oob_score} parameter')
        self.oob_metric_ = oob_score
        self.oob_score_: float = 0.

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.use_multiprocessing = use_multiprocessing

        self.leafs_cnt: int = 0
        self.fi: Dict[str, Union[int, float]] = defaultdict(int)
        self._estimators: List[MyTree] = []
        self._TreeClass: Type[MyTree] = MyTree
        self._tree_params_name_default: Dict[str, Any] = {}

    @staticmethod
    def _get_oob_score(metric: str, oob_predictions: Dict[str, List[float]], y: pd.Series) -> float:

        y_hat = pd.Series(np.empty(len(oob_predictions), float), index=list(oob_predictions.keys()))
        for idx, predictions in oob_predictions.items():
            y_hat[idx] = np.mean(predictions)

        if metric == 'mae':
            return np.mean(abs(y_hat - y))
        elif metric == 'rmse':
            return np.sqrt(np.mean((y_hat - y) ** 2))
        elif metric == 'mse':
            return np.mean((y_hat - y) ** 2)
        elif metric == 'mape':
            return 100 * np.mean(abs((y - y_hat) / y))
        elif metric == 'r2':
            return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    @staticmethod
    def _multiprocessing_initialiser(X: pd.DataFrame, y: pd.Series, TreeClass: Type[MyTree],
                                     cols_smpl_cnt: int, rows_smpl_cnt: int, metric: str, random_state: int = 42):
        MyForest._X = X
        MyForest._y = y
        MyForest._TreeClass = TreeClass
        MyForest._cols_smpl_cnt = cols_smpl_cnt
        MyForest._rows_smpl_cnt = rows_smpl_cnt
        random.seed(random_state)

    @staticmethod
    def _fit_estimator(estimator_i: int) -> Tuple[MyTree, int, pd.Series]:
        cols_idx = random.sample(range(len(MyForest._X.columns)), MyForest._cols_smpl_cnt)
        rows_idx = random.sample(range(len(MyForest._X)), MyForest._rows_smpl_cnt)
        estimator = MyForest._TreeClass()
        oob_data = MyForest._X.drop(rows_idx, axis=0).iloc[:, cols_idx]
        sub_X = MyForest._X.iloc[rows_idx, cols_idx]
        sub_y = MyForest._y.loc[sub_X.index]
        estimator.fit(sub_X, sub_y, len(MyForest._X))
        return estimator, estimator.leafs_cnt, estimator.predict(oob_data)

    @staticmethod
    def fit_multiprocessing(X: pd.DataFrame, y: pd.Series, n_estimators: int, cols_smpl_cnt: int, rows_smpl_cnt: int,
                            TreeClass: partial[MyTree], metric: str, use_multiprocessing: bool = False,
                            random_state: int = 42):
        oob_dict = defaultdict(list)
        general_fi: Dict[str, Union[int, float]] = defaultdict(int)
        estimators = []
        general_leafs_cnt = 0
        with Pool(cpu_count() if use_multiprocessing else 1,
                  MyForest._multiprocessing_initialiser, [X, y, TreeClass, cols_smpl_cnt, rows_smpl_cnt, metric,
                                                          random_state]
                  ) as pool:
            for estimator, leafs_cnt, oob_prediction in pool.imap_unordered(MyForest._fit_estimator,
                                                                            range(n_estimators)):
                estimators.append(estimator)
                general_leafs_cnt += leafs_cnt
                for idx, prediction in oob_prediction.items():
                    oob_dict[idx].append(prediction)
                for column, importance in estimator.fi.items():
                    general_fi[column] += importance
        oob_score = MyForest._get_oob_score(metric, oob_dict, y.loc[list(oob_dict.keys())])
        return general_leafs_cnt, estimators, general_fi, oob_score

    def fit(self, X: pd.DataFrame, y: pd.Series):
        cols_smpl_cnt = round(self.max_features * len(X.columns))
        rows_smpl_cnt = round(self.max_samples * len(X))
        tree_params = {param: vars(self).get(param, default)
                       for param, default in self._tree_params_name_default.items()}
        TreeClass = partial(self._TreeClass, **tree_params)
        self.leafs_cnt, self._estimators, self.fi, self.oob_score_ = (
            self.fit_multiprocessing(X, y, self.n_estimators, cols_smpl_cnt, rows_smpl_cnt,
                                     TreeClass, self.oob_metric_, self.use_multiprocessing, self.random_state)
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        results = []
        for estimator in self._estimators:
            results.append(estimator.predict(X))
        return reduce(lambda x, y: x + y, results) / len(results)

    def __str__(self):
        return (f'{self.__class__.__name__} class: n_estimators={self.n_estimators}, max_features={self.max_features},'
                f' max_samples={self.max_samples}, max_depth={self.max_depth},'
                f' min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}, bins={self.bins},'
                f' random_state={self.random_state}')


class MyForestReg(MyForest):
    def __init__(self, n_estimators: int = 10, max_features: float = 0.5, max_samples: float = 0.5,
                 random_state: int = 42,
                 max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = 16,
                 oob_score: Optional[str] = 'mae', use_multiprocessing: bool = False):
        super().__init__(n_estimators, max_features, max_samples, random_state,
                         max_depth, min_samples_split, max_leafs, bins, oob_score, use_multiprocessing)
        self._TreeClass = MyTreeReg
        for param_name, param in inspect.signature(self._TreeClass.__init__).parameters.items():
            if param_name != 'self':
                default = None
                if param.default is not param.empty:
                    default = param.default
                self._tree_params_name_default[param_name] = default

from data.reg import diab_X, diab_y

if __name__ == "__main__":
    forest = MyForestReg(100, max_depth=14, use_multiprocessing=True)
    forest.fit(diab_X, diab_y)
