import inspect
import random

from abc import ABC
from collections import defaultdict
from copy import deepcopy
from functools import reduce
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Type, Union

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


class MyKNN(ABC):
    def __init__(self, k: int = 3, metric: str = 'euclidean', weight: str = 'uniform'):
        self.k = k
        if metric not in ['euclidean', 'manhattan', 'chebyshev', 'cosine']:
            raise ValueError(f'Unknown ``metric``={metric} parameter')
        self.metric = metric
        if weight not in ['uniform', 'rank', 'distance']:
            raise ValueError(f'Unknown ``weight``={weight} parameter')
        self.weight = weight
        self.train_size: Optional[Tuple[int, int]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape

    def _get_distance(self, x1: pd.Series, x2: pd.Series) -> float:
        if self.metric == 'euclidean':
            return np.linalg.norm(x1 - x2)
        elif self.metric == 'manhattan':
            return np.linalg.norm(x1 - x2, ord=1)
        elif self.metric == 'chebyshev':
            return max(np.abs(x1 - x2))
        elif self.metric == 'cosine':
            return 1 - x1 @ x2 / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return np.inf

    def _get_k_neighbours(self, row: pd.Series) -> pd.DataFrame:
        distance_df = self.X_train.apply(
            self._get_distance, axis=1, x2=row, result_type='reduce').to_frame('distance')
        distance_label_df = distance_df.assign(label=self.y_train)
        nearest_neighbours = distance_label_df.nsmallest(self.k, 'distance')
        return nearest_neighbours

    def _get_weights(self, neighbours_df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def _get_label(self, row: pd.Series) -> Union[int, float]:
        raise NotImplementedError()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(self._get_label, axis=1)

    def __str__(self):
        return (f'{self.__class__.__name__} class: k={self.k}, metric={self.metric},'
                f'weight={self.weight}, train_size={self.train_size}')


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


class MyTreeReg(MyTree):
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20,
                 criterion: str = 'mse', bins: Optional[int] = None):
        possible_criterions = ['mse', 'variance']
        if criterion not in possible_criterions:
            raise ValueError('Unknown gain function option in ``criterion`` parameter.'
                             f' Should be one of: [{", ".join(possible_criterions)}]')
        super().__init__(max_depth, min_samples_split, max_leafs, criterion, bins)


class MyBagging(ABC):
    def __init__(self, estimator: Optional[Union[MyLinear, MyKNN, MyTree]] = None, n_estimators: int = 10,
                 max_samples: float = 1., random_state: int = 42, oob_score: Optional[str] = None):
        if estimator is None:
            raise ValueError("Can't train the model without known base estimator")
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_metric_ = oob_score
        self.oob_score_: float = 0.

        self.estimators: List[Union[MyLinear, MyKNN, MyTree]] = []
        self.fi: Dict[str, Union[int, float]] = defaultdict(int)

    def _get_oob_score(self, oob_predictions: Dict[str, List[float]], y: pd.Series) -> float:
        raise NotImplementedError()

    def _predict_oob(self, estimator: Union[MyLinear, MyKNN, MyTree], oob_dict: Dict[Hashable, List[float]],
                     data: pd.DataFrame):
        for idx, prediction in estimator.predict(data).items():
            oob_dict[idx].append(prediction)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        rows_smpl_cnt = round(self.max_samples * len(X))
        random.seed(self.random_state)
        # oob_dict = defaultdict(list)
        rows_idxs = [random.choices(range(len(X)), k=rows_smpl_cnt) for i in range(self.n_estimators)]
        for rows_idx in rows_idxs:
            estimator = deepcopy(self.estimator)
            # oob_data = X.drop(rows_idx, axis=0).iloc[:, cols_idx]
            sub_X = X.iloc[rows_idx]
            sub_y = y.loc[sub_X.index]
            if isinstance(estimator, MyTree):
                estimator.fit(sub_X, sub_y, len(X))
            else:
                estimator.fit(sub_X, sub_y)
            self.estimators.append(estimator)
            # self._predict_oob(estimator, oob_dict, oob_data)
        # self.oob_score_ = self._get_oob_score(oob_dict, y.loc[list(oob_dict.keys())])
        # for estimator in self._estimators:
        #     for column, importance in estimator.fi.items():
        #         self.fi[column] += importance

    def predict(self, X: pd.DataFrame) -> pd.Series:
        results = []
        for estimator in self.estimators:
            results.append(estimator.predict(X))
        return reduce(lambda x, y: x + y, results) / len(results)

    def __str__(self):
        return (f'{self.__class__.__name__} class: estimator={self.estimator}, n_estimators={self.n_estimators},'
                f' max_samples={self.max_samples}, random_state={self.random_state}')


class MyBaggingReg(MyBagging):
    def __init__(self, estimator: Optional[Union[MyLinear, MyKNN, MyTree]] = None, n_estimators: int = 10,
                 max_samples: float = 1., random_state: int = 42, oob_score: Optional[str] = 'mse'):
        possible_oob_scores = ['mae', 'rmse', 'mse', 'mape', 'r2']
        if oob_score not in possible_oob_scores:
            raise ValueError(f'Unknown ``oob_score``={oob_score} parameter'
                             f' Should be one of: [{", ".join(possible_oob_scores)}]')
        super().__init__(estimator, n_estimators, max_samples, random_state, oob_score)

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
