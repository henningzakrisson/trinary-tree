""" Common functions for the binary- and trinary trees"""
import numpy as np
import pandas as pd
import itertools
import warnings
from src.trinary_tree.common.custom_exceptions import MissingValuesInResponse
from src.trinary_tree.common.custom_warnings import (
    MissingFeatureWarning,
    ExtraFeatureWarning,
)


def fix_datatypes(X, y=None, w=None):
    """Make sure datasets are pandas DataFrames and Series"""
    X = pd.DataFrame(X).copy() if isinstance(X, np.ndarray) else X.copy()
    for feature in X:
        if X[feature].dtype == "int":
            X[feature] = X[feature].astype(float)
    if y is not None:
        y = pd.Series(y).copy() if isinstance(y, np.ndarray) else y.copy()
        if y.dtype == "float" or y.dtype == "int":
            y = y.astype(float)

        if np.any(y.isna()) or np.any(y.isna()):
            raise MissingValuesInResponse("n/a not allowed in response (y)")

    if w is not None:
        w = pd.Series(w).copy() if isinstance(w, np.ndarray) else w.copy()
        if w.dtype in ["float", "int"]:
            w = w.astype(float)

    return X, y, w


def fit_response(y, categories=None, w=None):
    """Get the response estimate given this set of responses"""

    if w is None:
        w = pd.Series(index=y.index, dtype=float)
        w.loc[:] = 1

    if y.dtype == "float" or y.dtype == "int":
        y_hat = (w * y).sum() / w.sum()
        y_prob = {}
        categories = None
    else:
        if categories is None:
            categories = list(y.unique())
        y_prob = {
            category: sum((y == category) * w) / sum(w) for category in categories
        }
        y_hat = max(y_prob, key=y_prob.get)

    return y_hat, y_prob, categories


def calculate_loss(y, y_hat=None, y_prob=None, w=None):
    """Calculate the loss of the response set

    Gini if classification problem, sse if regression

    Args:
        y: response pd.Series
        y_hat: response estimate. If None, will be calculated as mean/mode
        w: Node membership weight

    Returns:
        loss as a float
    """
    if len(y) == 0:
        return 0

    if w is None:
        w = pd.Series(index=y.index, dtype=float)
        w.loc[:] = 1

    if y.dtype == "float" or y.dtype == "int":
        if y_hat is None:
            y_hat, _, _ = fit_response(y, w=w)
        return calculate_mse(y, y_hat=y_hat, w=w)
    else:
        if y_prob is None:
            _, y_prob, _ = fit_response(y, w=w)
        return calculate_xe(y, y_prob=y_prob, w=w)


def calculate_mse(y, y_hat, w):
    """Calculate mean squared error"""
    return (w * (y - y_hat).pow(2)).sum() / w.sum()


def calculate_xe(y, y_prob, w):
    """Calculate cross entropy"""
    eps = 1e-30
    if isinstance(y_prob, dict):
        # This is for one probability prediction (a dict)
        ps = y.replace(y_prob)
        xes = -w * np.log(ps + eps)
    else:
        # This is for predicting for an entire set (a series)
        idx, cols = pd.factorize(y)
        ps = y_prob.reindex(cols, axis=1).to_numpy()[np.arange(len(y_prob)), idx]
        xes = -w * np.log(ps + eps)
    return sum(xes)


def check_terminal_node(tree):
    """ " Check if pruning conditions are fulfilled"""
    return (tree.depth >= tree.max_depth) or (tree.n <= tree.min_samples_leaf)


def get_splitter_candidates(x):
    """Get potential candidates for splitters

    For continous variables, all values that split the test_data in a unique way is found by looking at
     values between all unique non-missing datapoints. For categorical variables, all possible
     ways to split the set of unique categories are tried.

    Args:
        x: Covariate vector for one certain feature as a pandas Series

    Returns:
        numpy array or list of relevant thresholds
    """
    if np.all(x.isna()):
        return []
    elif x.dtype == "float":
        values = x.drop_duplicates()
        return values.sort_values().rolling(2).mean().dropna().values
    elif x.dtype == "object":
        values = x.dropna().unique()
        left_sets = list(
            itertools.chain.from_iterable(
                itertools.combinations(values, r) for r in range(1, len(values))
            )
        )
        right_sets = [
            [value for value in values if value not in left_set]
            for left_set in left_sets
        ]
        return [
            {"left": left_set, "right": right_set}
            for left_set, right_set in zip(left_sets, right_sets)
        ]


def get_indices(x, splitter, default_split="none"):
    """Get left and right indices given a splitter"""
    if x.dtype == "float":
        index_left = x < splitter
        index_right = x >= splitter
    else:
        index_left = x.isin(splitter["left"])
        index_right = x.isin(splitter["right"])
    if default_split == "left":
        index_left |= ~index_left & ~index_right
    elif default_split == "right":
        index_right |= ~index_left & ~index_right

    return index_left, index_right


def get_feature_importance(tree):
    """Calculate feature importance for all features in X

    Return:
        dict with keys corresponding to feature and values corresponding to their feature importances. Sums to 1.
    """
    node_importances = tree._get_node_importances(
        node_importances={feature: [] for feature in tree.features}
    )
    total_importances = {
        feature: sum(node_importances[feature]) for feature in node_importances
    }
    if sum(total_importances.values()) == 0:
        feature_importances = {feature: 0 for feature in node_importances}
    else:
        feature_importances = {
            feature: total_importances[feature] / sum(total_importances.values())
            for feature in total_importances
        }
    return feature_importances


def check_features(X, features):
    """Check so that all relevant features are available and none are redundant. Return fixed X"""
    missing_features = [feature for feature in features if feature not in X.columns]
    if len(missing_features) > 0:
        warnings.warn(
            f"Covariate matrix missing features {missing_features} - filling with n/a",
            MissingFeatureWarning,
        )
        for feature in missing_features:
            X[feature] = np.nan

    extra_features = [feature for feature in X.columns if feature not in features]
    if len(extra_features) > 0:
        warnings.warn(
            f"Covariate matrix missing features {extra_features} - filling with n/a",
            ExtraFeatureWarning,
        )
        X = X.drop(extra_features, axis=1)

    return X
