import itertools
import warnings
import pandas as pd
import numpy as np
from src.trinary_tree.common.custom_exceptions import (
    MissingValuesInResponse,
    CantPrintUnfittedTree,
)
from .common.custom_warnings import (
    MissingFeatureWarning,
    ExtraFeatureWarning,
)
from .common.functions import (
    fix_datatypes,
    fit_response,
    calculate_loss,
    check_terminal_node,
    get_splitter_candidates,
    get_indices,
    check_features,
)


class WeightedTree:
    """Module for classification and regression trees with standard handling missing test_data

    The missing test_data points are assigned weights as to which branch to join
    """

    def __init__(
        self,
        min_samples_leaf=20,
        max_depth=2,
        depth=0,
        categories=None,
    ):
        """Initiate the tree

        Args:
            min_samples_leaf: number of datapoints as minimum to allow for daughter nodes (-1)
            max_depth: number of levels allowed in the tree
            depth: current depth. root node has depth 0
            missing_rule: strategy to handle missing values
            categories: Possible values of response - learnt from original dataset

        Returns:
            Tree object (which is a node. Can be a root node, a daughter node and/or a terminal node).
        """
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.depth = depth
        self.categories = categories
        self.n = 0
        self.y_hat = None
        self.y_prob = {}
        self.loss = None
        self.feature = None
        self.feature_type = None
        self.response_type = None
        self.features = []
        self.splitter = None
        self.default_split = None
        self.left = None
        self.right = None
        self.p_left = 1
        self.p_right = 0
        self.node_importance = 0

    def fit(self, X, y, w=None):
        """Recursive method to fit the decision tree.

        Will call itself to create daughter nodes if applicable.

        Args:
            X: covariate vector (n x p). numpy array or pandas DataFrame.
            y: response vector (n x 1). numpy array or pandas Series.
            w: node membership weight vector (n x 1). numpy array or pandas Series.

        Raises:
            MissingValuesInResponse: Can not fit to missing categories, thus errors out
        """
        X, y, w = fix_datatypes(X, y, w)
        if w is None:
            w = pd.Series(index=y.index, dtype=float)
            w.loc[:] = 1

        self.features = X.columns

        self.n = w.sum()

        self.y_hat, self.y_prob, self.categories = fit_response(y, self.categories, w)
        if self.categories is not None:
            self.response_type = "object"
        else:
            self.response_type = "float"

        self.loss = calculate_loss(y=y, w=w)

        # Check pruning conditions
        if check_terminal_node(self):
            self.left, self.right = None, None
            return

        # Find splitting parameters
        self.feature, self.splitter = self._find_split(X, y, w)

        if self.feature is None:
            self.left, self.right = None, None
            return

        self.feature_type = "float" if X[self.feature].dtype == "float" else "object"

        index_left, index_right = get_indices(
            X[self.feature], self.splitter, self.default_split
        )

        # Node weights
        self.p_left, self.p_right = self._get_split_probabilities(
            index_left, index_right
        )
        w_left, w_right = w.copy(), w.copy()
        index_na = (~index_left) & (~index_right)
        w_left.loc[index_na] *= self.p_left
        w_right.loc[index_na] *= self.p_right
        index_left |= index_na
        index_right |= index_na

        # Send test_data to daughter nodes
        self.left, self.right = self._initiate_daughter_nodes()
        self.left.fit(X.loc[index_left], y.loc[index_left], w_left.loc[index_left])
        self.right.fit(X.loc[index_right], y.loc[index_right], w_right.loc[index_right])

        self.node_importance = self._calculate_importance()

    def _get_split_probabilities(self, index_left, index_right):
        n_left, n_right = index_left.sum(), index_right.sum()
        p_left, p_right = n_left / (n_left + n_right), n_right / (n_left + n_right)
        return p_left, p_right

    def _find_split(self, X, y, w) -> tuple:
        """Calculate the best split for a decision tree

        Args:
            X: Covariates to choose from
            y: response to fit nodes to
            w: node weights

        Returns:
            best_feature: feature to split by for minimum loss
            best_splitter: threshold or left-category-set to split feature by for minimum loss
        """
        # Initiate here in order to not grow more if this loss is not beaten
        loss_best = self.loss
        best_feature, best_splitter = None, None

        features = [
            feature for feature in X.columns if X[feature].isna().sum() < len(X)
        ]
        for feature in features:
            splitters = get_splitter_candidates(X[feature])
            for splitter in splitters:
                loss = self._calculate_split_loss(X, y, w, feature, splitter)
                if loss < loss_best:
                    loss_best = loss
                    best_feature, best_splitter = (
                        feature,
                        splitter,
                    )

        return best_feature, best_splitter

    def _calculate_split_loss(self, X, y, w, feature, splitter):
        """Calculates the sum of squared errors for this split

        Args:
            X: covariate vector
            y: response vector
            w: node weights
            feature: feature of X to split test_data on
            splitter: threshold or set of categories that will go to the left node

        Returns:
            Total loss of this split for all daughter nodes
        """
        index_left, index_right = get_indices(X[feature], splitter)
        p_left, p_right = self._get_split_probabilities(index_left, index_right)
        w_left, w_right = w.copy(), w.copy()
        index_na = (~index_left) & (~index_right)
        w_left.loc[index_na] *= p_left
        w_right.loc[index_na] *= p_right
        index_left |= index_na
        index_right |= index_na

        # To avoid hyperparameter-illegal splits
        if (w.loc[index_left].sum() < self.min_samples_leaf) or (
            w.loc[index_right].sum() < self.min_samples_leaf
        ):
            return self.loss

        loss_left_weighted = (
            calculate_loss(y=y.loc[index_left], w=w_left.loc[index_left])
            * w_left.loc[index_left].sum()
        )
        loss_right_weighted = (
            calculate_loss(y=y.loc[index_right], w=w_right.loc[index_right])
            * w_right.loc[index_right].sum()
        )

        return (loss_left_weighted + loss_right_weighted) / self.n

    def _initiate_daughter_nodes(self):
        """Create daughter nodes

        Return:
            tuple of two Trees. The one in the middle is None for non-trinary trees.
        """
        left = WeightedTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            categories=self.categories,
        )
        right = WeightedTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            categories=self.categories,
        )

        return left, right

    def _calculate_importance(self):
        """ "Calculate node importance for the split in this node

        Return:
            Node importance as a float
        """
        # If no values of the training test_data actually end up here it is of no importance
        if self.n == 0:
            importance = 0
        else:
            importance = (
                self.loss
                - (self.left.n * self.left.loss + self.right.n * self.right.loss)
                / self.n
            )
        return importance

    def _get_node_importances(self, node_importances):
        """Get node importances for this node and all its daughters

        Args:
            node_importances: dict with keys corresponding to feature and values corresponding to their node importances.

        Return:
            dict with keys corresponding to feature and values corresponding to their feature importances.
        """
        if self.feature is not None:
            node_importances[self.feature].append(self.node_importance)
        if self.left is not None:
            node_importances = self.left._get_node_importances(
                node_importances=node_importances
            )
            node_importances = self.right._get_node_importances(
                node_importances=node_importances
            )
        return node_importances

    def predict(self, X, prob=False):
        """Recursive method to predict from new of features

        Args:
            Covariate vector X (m x p) of same secondary dimension as training covariate vector
            prob: True if predict probabilities rather than responses

        Returns:
            response predictions y_hat as a pandas Series. DataFrame if probabilities.
        """
        X, _, _ = fix_datatypes(X)
        X = check_features(X, self.features)

        if self.response_type == "object":
            y_prob = pd.DataFrame(index=X.index, columns=self.categories, dtype=float)
            if self.left is None:
                for category in self.categories:
                    y_prob[category] = self.y_prob[category]
            else:
                index_left, index_right = get_indices(X[self.feature], self.splitter)
                index_na = (~index_left) & (~index_right)

                y_prob.loc[index_left] = self.left.predict(X.loc[index_left], prob=True)
                y_prob.loc[index_right] = self.right.predict(
                    X.loc[index_right], prob=True
                )
                y_prob.loc[index_na] = self.p_left * self.left.predict(
                    X.loc[index_na], prob=True
                ) + self.p_right * self.right.predict(X.loc[index_na], prob=True)

        else:
            y_hat = pd.Series(index=X.index, dtype=self.response_type)
            if self.left is None:
                y_hat.loc[:] = self.y_hat
            else:
                index_left, index_right = get_indices(X[self.feature], self.splitter)
                index_na = (~index_left) & (~index_right)

                y_hat.loc[index_left] = self.left.predict(X.loc[index_left])
                y_hat.loc[index_right] = self.right.predict(X.loc[index_right])
                y_hat.loc[index_na] = self.p_left * self.left.predict(
                    X.loc[index_na]
                ) + self.p_right * self.right.predict(X.loc[index_na])

        if prob:
            return y_prob
        elif self.response_type == "float":
            return y_hat
        else:  # categorical prediction
            return y_prob.idxmax(axis=1)

    def print(self):
        """Print the tree structure"""
        if self.y_hat is None:
            raise CantPrintUnfittedTree("Can't print tree before fitting to test_data")

        hspace = "---" * self.depth
        print(hspace + f"Number of observations: {np.round(self.n,2)}")
        if isinstance(self.y_hat, float):
            print(hspace + f"Response estimate: {np.round(self.y_hat,2)}")
        else:
            print(hspace + f"Response estimate: {self.y_hat}")
        print(hspace + f"loss: {np.round(self.loss,2)}")
        if self.left is not None:
            if self.feature_type == "float":
                left_rule = f"if {self.feature} <  {np.round(self.splitter,2)}"
                right_rule = f"if {self.feature} >=  {np.round(self.splitter,2)}"
            elif self.feature_type == "object":
                left_rule = f"if {self.feature} is " + ", ".join(self.splitter["left"])
                right_rule = f"if {self.feature} is " + ", ".join(
                    self.splitter["right"]
                )
            print(hspace + f"{left_rule}:")
            self.left.print()
            print(hspace + f"{right_rule}:")
            self.right.print()


if __name__ == "__main__":
    """Main function to make the file run- and debuggable."""
    from src.common.functions import get_feature_importance

    folder_path = "/tests/test_data"
    df_train = pd.read_csv(f"{folder_path}/train_data_weighted_cat.csv", index_col=0)
    X_train = df_train.drop("y", axis=1)
    y_train = df_train["y"]

    df_test = pd.read_csv(f"{folder_path}/test_data_weighted_cat.csv", index_col=0)
    X_test = df_test[["number", "fruit"]]
    y_prob = df_test[["bad", "good", "great"]]
    y_test = df_test["y"]

    tree = WeightedTree(max_depth=2, min_samples_leaf=1)
    tree.fit(X_train, y_train)

    y_prob_hat = tree.predict(X_test, prob=True)
    y_hat = tree.predict(X_test)
