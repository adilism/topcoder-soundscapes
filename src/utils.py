import os
import random
from collections import Counter, defaultdict
from typing import Generator, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder


def metric(
    labels: np.ndarray,
    preds: np.ndarray,
    class_metric: bool = False,
    normalize: bool = False,
) -> Union[List, np.number]:
    """ROC AUC metric to use with Pytorch Lightning"""
    n_labels = labels.shape[1]
    if normalize:
        preds = preds / np.sum(preds, axis=1, keepdims=True)
    res = [roc_auc_score(labels[:, c], preds[:, c]) for c in range(n_labels)]
    return res if class_metric else np.mean(res)


def get_group_folds(
    train: pd.DataFrame,
    n_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = 4,
    stratified: bool = True,
) -> List:
    g = train.groupby("group")["labels"].first()
    if stratified:
        kf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        kf = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    folds = []
    for train_idx, valid_idx in kf.split(g, g):
        train_idx = train.index[train["group"].isin(train_idx)]
        valid_idx = train.index[train["group"].isin(valid_idx)]
        folds.append((train_idx, valid_idx))
    return folds


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


def stratified_group_k_fold(
    y: np.ndarray, groups: Union[List, pd.Series], k: int, seed: int = None
) -> Generator:
    """StratifiedKFold that takes groups into account and return balanced split
    Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
    """
    if seed is not None:
        raise ValueError("Not sure what seed does in the original implementation")

    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_stratified_group_folds(
    train: pd.DataFrame,
    n_splits: int = 5,
    stratified: bool = True,  # unused, but left for easy drop-in replacement
    random_state: int = None,
) -> List:
    enc = LabelEncoder()
    y = enc.fit_transform(train["labels"])

    folds = []
    kf = stratified_group_k_fold(y, train["group"], k=n_splits, seed=random_state)
    for train_idx, valid_idx in kf:
        train_idx = train.iloc[train_idx].index
        valid_idx = train.iloc[valid_idx].index
        folds.append((train_idx, valid_idx))
    return folds
