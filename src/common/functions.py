""" Common functions for the binary- and trinary trees"""
import numpy as np
import pandas as pd
import itertools
import warnings
from src.common.custom_exceptions import MissingValuesInResponse
from src.common.custom_warnings import MissingFeatureWarning, ExtraFeatureWarning


def fix_datatypes(X, y=None):
    """Make sure datasets are pandas DataFrames and Series"""
    X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
    for feature in X:
        if X[feature].dtype == "int":
            X[feature] = X[feature].astype(float)
    if y is not None:
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        if y.dtype in ["float", "int"]:
            y = y.astype(float)

        if np.any(y.isna()) or np.any(y.isna()):
            raise MissingValuesInResponse("n/a not allowed in response (y)")

    return X, y


def fit_response(y, categories=None):
    """Get the response estimate given this set of responses"""

    if y.dtype == "float":
        y_hat = y.mean()
        y_prob = {}
        categories = None
    else:
        if categories is None:
            categories = list(y.unique())
        y_prob = (y.value_counts() / len(y)).to_dict()
        y_hat = max(y_prob, key=y_prob.get)
        for category in categories:
            if category not in y_prob:
                y_prob[category] = 0

    return y_hat, y_prob, categories


def calculate_loss(y, y_hat=None):
    """Calculate the loss of the response set

    Gini if classification problem, sse if regression

    Args:
        y: response pd.Series
        y_hat: response estimate. If None, will be calculated as mean/mode

    Returns:
        loss as a float
    """
    if len(y) == 0:
        return 0
    elif y.dtype == "float":
        y_hat = y.mean() if not y_hat else y_hat
        return (y - y_hat).pow(2).mean()
    else:
        ps = [(y == y_value).mean() for y_value in y.unique()]
        return sum([p * (1 - p) for p in ps])


def check_terminal_node(tree):
    """ " Check if pruning conditions are fulfilled"""
    return (tree.depth >= tree.max_depth) or (tree.n <= tree.min_samples_leaf)


def get_splitter_candidates(x):
    """Get potential candidates for splitters

    For continous variables, all values that split the data in a unique way is found by looking at
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


def get_indices(x, splitter, default_split="middle"):
    """Get left and right indices given a splitter"""
    if x.dtype == "float":
        index_left = x < splitter
        index_right = x >= splitter
    elif x.dtype == "object":
        index_left = x.isin(splitter["left"])
        index_right = x.isin(splitter["right"])
    if default_split == "left":
        index_left |= x.isna()
    elif default_split == "right":
        index_right |= x.isna()

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
