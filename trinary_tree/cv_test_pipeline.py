import copy

import numpy as np
import pandas as pd
import logging

from .binary_tree import BinaryTree
from .trinary_tree import TrinaryTree
from .trinary_mia_tree import TrinaryMiaTree
from .weighted_tree import WeightedTree
from .common.functions import calculate_loss
from sklearn.model_selection import StratifiedKFold, KFold


def create_MCAR_Xs(X, ps, seed=None):
    """
    Takes a pandas DataFrame X and a list of probabilities ps, and returns a dictionary
    of modified versions of X, with values randomly set to NaN based on the corresponding
    probabilities in ps.
    """
    rng = np.random.default_rng(seed)
    Xs = {}
    n = len(X)

    X = X.copy()
    n_to_drop = []
    for i, p in enumerate(ps):
        n_to_drop += [int(p * n) - sum(n_to_drop)]
        for j in X.columns:
            to_remove = np.random.choice(
                X.loc[~X[j].isna(), j].index, size=n_to_drop[i], replace=False
            )
            X.loc[to_remove, j] = np.nan
        Xs[p] = X.copy()
    return Xs


def create_IM_Xs(X, ps, seed=None):
    """
    Takes a pandas DataFrame X and a list of probabilities ps, and returns a dictionary
    of modified versions of X, with values randomly set to NaN based on the corresponding
    probabilities in ps, with the largest values always being dropped first.
    """
    rng = np.random.default_rng(seed)
    Xs = {}
    n = len(X)

    X = X.copy()
    n_to_drop = []
    for i, p in enumerate(ps):
        n_to_drop += [int(p * n) - sum(n_to_drop)]
        for j in X.columns:
            available_to_drop = X.loc[~X[j].isna(), j].index
            size_order = np.argsort(X.loc[available_to_drop, j])
            to_remove = available_to_drop[size_order[: n_to_drop[i]]]
            X.loc[to_remove, j] = np.nan
        Xs[p] = X.copy()
    return Xs


def split_dataset_into_folds(X, y, n_folds, seed=None):
    if y.dtype == "float" or y.dtype == "int":
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = {}
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold = {i: {"Train": (X_train, y_train), "Test": (X_test, y_test)}}
        folds.update(fold)

    return folds


def split_missing_datasets_into_folds(Xs, y, n_folds, seed=None):
    missing_folds = {
        p: split_dataset_into_folds(Xs[p], y, n_folds=n_folds, seed=seed) for p in Xs
    }
    return missing_folds


def tune_max_depth(folds, max_max_depth=10, min_samples_leaf=20):
    max_depths = np.arange(max_max_depth + 1)
    losses = pd.Series(index=max_depths, dtype=float)

    for max_depth in max_depths:
        logging.info(f"Testing max_depth = {max_depth}")
        tree = BinaryTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        losses[max_depth] = calculate_cv_loss(folds=folds, tree=tree)

        if losses.iloc[max_depth] >= losses.iloc[max_depth - 1]:
            break

    return losses.idxmin()


def setup_equal_trees(max_depth=None, min_samples_leaf=None, tree_types="all"):
    trees = {
        "Majority": BinaryTree(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            missing_rule="majority",
        ),
        "MIA": BinaryTree(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, missing_rule="mia"
        ),
        "Weighted": WeightedTree(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        ),
        "Trinary": TrinaryTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf),
        "TrinaryMia": TrinaryMiaTree(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        ),
    }

    if tree_types == "all":
        return trees

    return {tree_type: trees[tree_type] for tree_type in tree_types}


def calculate_missing_cvs_loss(missing_folds, trees, missing_set):
    losses = pd.DataFrame(index=missing_folds.keys(), columns=trees.keys(), dtype=float)
    if missing_set == "test":
        fitted_trees = fit_non_missing_trees(missing_folds[0], trees)

    for p in missing_folds:
        logging.info(f"Calculating loss with missingness = {p}")
        for tree_type in trees:
            logging.info(f"Calculating loss for {tree_type.lower()} tree")
            if missing_set == "test":
                losses.loc[p, tree_type] = calculate_cv_loss(
                    missing_folds[p],
                    trees[tree_type],
                    fitted_trees=fitted_trees[tree_type],
                )
            else:
                losses.loc[p, tree_type] = calculate_cv_loss(
                    missing_folds[p], trees[tree_type]
                )

    return losses


def fit_non_missing_trees(folds, trees):
    fitted_trees = {}
    for tree_type in trees:
        logging.info(f"Pre_fitting trees for {tree_type.lower()} tree")
        fitted_trees[tree_type] = {}
        for i, fold in folds.items():
            logging.info(f"Fitting for fold {i}")
            X_train, y_train = fold["Train"]
            tree = trees[tree_type]
            tree.fit(X_train, y_train)
            fitted_trees[tree_type][i] = copy.deepcopy(tree)

    return fitted_trees


def calculate_cv_loss(folds, tree, fitted_trees=None):
    y = pd.concat([folds[fold]["Test"][1] for fold in folds]).sort_index()
    if y.dtype == "object":
        y_prob = pd.DataFrame(columns=y.unique(), index=y.index, dtype="float")
    else:
        y_hat = pd.Series(index=y.index, dtype="float")
    for i, fold in folds.items():
        logging.info(f"Calculating loss for fold {i+1}/{len(folds.items())}")
        if fitted_trees is None:
            X_train, y_train = fold["Train"]
            tree.fit(X_train, y_train)
        else:
            tree = fitted_trees[i]
        X_test, _ = fold["Test"]
        if y.dtype == "object":
            # This is done via update + fillna since there is no guarantee that all classes are in all folds
            # Then missing classes are assigned probability 0
            y_prob.update(tree.predict(X_test, prob=True))
            y_prob.loc[X_test.index].fillna(0, inplace=True)
        else:
            y_hat.loc[X_test.index] = tree.predict(X_test)

    if y.dtype == "object":
        return calculate_loss(y=y, y_prob=y_prob)
    else:
        return calculate_loss(y=y, y_hat=y_hat)


def calculate_dataset_missing_losses(
    data_set,
    data_folder,
    seed_missingness,
    seed_fold_split,
    ps,
    n_folds,
    min_samples_leaf,
    max_max_depth,
    tree_types,
    missing_set,
    missingness="MCAR",
):
    logging.info(f"Pre-processing {data_set} data")
    X = pd.read_csv(f"{data_folder}/{data_set}.csv", index_col=0)
    y = X.pop("y")

    if missingness == "MCAR":
        Xs = create_MCAR_Xs(X, ps, seed=seed_missingness)
    elif missingness == "IM":
        Xs = create_IM_Xs(X, ps, seed=seed_missingness)
    missing_folds = split_missing_datasets_into_folds(
        Xs, y, n_folds, seed=seed_fold_split
    )
    max_depth = tune_max_depth(
        missing_folds[0],
        max_max_depth=max_max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    logging.info(f"{data_set} max_depth set to {max_depth}")
    trees = setup_equal_trees(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        tree_types=tree_types,
    )
    return calculate_missing_cvs_loss(missing_folds, trees, missing_set)


def calculate_loss_for_files(
    data_sets,
    data_folder,
    seed_missingness,
    seed_fold_split,
    ps,
    n_folds,
    min_samples_leaf,
    max_max_depth,
    tree_types,
    output_data_folder=None,
    missing_set="all",
    missingness="MCAR",
):
    log_folder = "/home/heza7322/PycharmProjects/missing-value-handling-in-carts/logs"
    logging.basicConfig(level=logging.INFO)
    losses = {}
    for data_set in data_sets:
        losses[data_set] = calculate_dataset_missing_losses(
            data_set,
            data_folder,
            seed_missingness,
            seed_fold_split,
            ps,
            n_folds,
            min_samples_leaf,
            max_max_depth,
            tree_types,
            missing_set=missing_set,
            missingness=missingness,
        )
        if output_data_folder is not None:
            losses[data_set].to_csv(
                f"{output_data_folder}/{missingness}_{missing_set}/{data_set}.csv"
            )
    return pd.concat(losses, names=["data_set", "missingness"])


if __name__ == "__main__":
    # Set up variables
    input_data_folder = (
        "/home/heza7322/PycharmProjects/missing-value-handling-in-carts/data/cleaned"
    )
    output_data_folder = (
        "/home/heza7322/PycharmProjects/missing-value-handling-in-carts/data/results"
    )
    data_sets = [
        "titanic",
        "auto_mpg",
        "life_expectancy",
        "lymphography",
        "seeds",
        "cement",
        "boston_housing",
        "black_friday",
    ]
    tree_types = ["Majority", "MIA", "Weighted", "Trinary", "TrinaryMia"]
    seed_missingness = 10
    seed_fold_split = 11
    ps = np.arange(10) / 10
    n_folds = 10
    min_samples_leaf = 20
    max_max_depth = 5
    missing_set = "all"
    missingness = "IM"

    losses = calculate_loss_for_files(
        data_sets=data_sets,
        data_folder=input_data_folder,
        seed_missingness=seed_missingness,
        seed_fold_split=seed_fold_split,
        ps=ps,
        n_folds=n_folds,
        min_samples_leaf=min_samples_leaf,
        max_max_depth=max_max_depth,
        tree_types=tree_types,
        output_data_folder=output_data_folder,
        missing_set=missing_set,
        missingness=missingness,
    )
