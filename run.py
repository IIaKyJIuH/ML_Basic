import numpy as np
import pandas as pd

from bagging_reg import MyBaggingReg
from data.clf import bank_X, bank_y, clf_X, clf_y
from data.reg import diab_X, diab_y, reg_X, reg_y
from decision_tree.decision_tree_reg import MyTreeReg


def roc_auc(y_true, y_pred):
    y, y_hat_probs = y_true, y_pred
    PN = y.sum() * (len(y) - y.sum())
    sorted_probs_arg = np.argsort(y_hat_probs)[::-1]
    sorted_labels = y[sorted_probs_arg]
    sorted_probs = y_hat_probs[sorted_probs_arg]
    positive_idxs = []
    result_sum = 0.
    for idx, (label, prob) in enumerate(zip(sorted_labels, sorted_probs)):
        if label == 0:
            pos_probs_before = sorted_probs[positive_idxs]
            equal_pos_probs_cnt = np.sum(pos_probs_before == prob)
            above_pos_probs_cnt = np.sum(pos_probs_before > prob)
            result_sum += equal_pos_probs_cnt / 2 + above_pos_probs_cnt
        else:
            positive_idxs.append(idx)
    return np.around(result_sum, 10) / PN

# tree = MyTreeReg(max_depth=2,
#                  min_samples_split=2,
#                  max_leafs=20,
#                  criterion='mse',
#                  bins=None)

# bagging = MyBaggingReg(estimator=tree,
#                        n_estimators=3,
#                        max_samples=0.2,
#                        random_state=42)

# bagging.fit(reg_X, reg_y)
# test_indices = [73, 18, 118, 78, 76, 31, 64, 141, 68, 82, 110, 12, 36,
#                 9, 19, 56, 104, 69, 55, 132]
# test_X = reg_X.loc[test_indices]
# predict = bagging.predict(test_X)
# print(predict.sum(), bagging.estimator.print_tree(bagging.estimator.root), tree.leafs_cnt)

tree = MyTreeReg(max_depth=15,
                 min_samples_split=35,
                 max_leafs=30)

tree.fit(diab_X, diab_y)
print(tree.print_tree(print_with_nums=True), tree.leafs_cnt)
