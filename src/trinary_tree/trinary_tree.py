import itertools
import warnings
import pandas as pd
import numpy as np
import warnings

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
    get_feature_importance,
)


class TrinaryTree:
    """Module for classification and regression trees with third-node handling of missing values

    The missing test_data strategy creates a third node for missing values, which inherits test_data and output from mother node
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
        self.node_importance = 0
        self.n_true = 0
        self.loss_true = None
        self.middle = None

    def fit(self, X, y, X_true=None, y_true=None):
        """Recursive method to fit the decision tree.

        Will call itself to create daughter nodes if applicable.

        Args:
            X: covariate vector (n x p). numpy array or pandas DataFrame.
            y: response vector (n x 1). numpy array or pandas Series.
            X_true: covariate vector (n x p) of test_data that ends up in this node
            during training.
            y_true: response vector (n x 1) of test_data that ends up in this node
            during training.

        Raises:
            MissingValuesInResponse: Can not fit to missing categories, thus errors out
        """
        X, y, _ = fix_datatypes(X, y)
        if X_true is None and y_true is None:
            X_true, y_true = X, y
        X_true, y_true, _ = fix_datatypes(X_true, y_true)
        self.features = X.columns

        self.n = len(y)
        self.n_true = len(y_true)

        self.y_hat, self.y_prob, self.categories = fit_response(y, self.categories)
        if self.categories is not None:
            self.response_type = "object"
        else:
            self.response_type = "float"

        self.loss = calculate_loss(y=y, y_hat=self.y_hat, y_prob=self.y_prob)
        self.loss_true = calculate_loss(y=y_true, y_hat=self.y_hat, y_prob=self.y_prob)

        # Check pruning conditions
        if check_terminal_node(self):
            self.left, self.middle, self.right = None, None, None
            return

        # Find splitting parameters
        self.feature, self.splitter = self._find_split(X, y)

        if self.feature is None:
            self.left, self.middle, self.right = None, None, None
            return

        self.feature_type = "float" if X[self.feature].dtype == "float" else "object"

        index_left, index_right = get_indices(X[self.feature], self.splitter)
        index_left_true, index_right_true = get_indices(
            X_true[self.feature], self.splitter
        )
        index_middle_true = (~index_left_true) & (~index_right_true)

        # Send test_data to daughter nodes
        self.left, self.middle, self.right = self._initiate_daughter_nodes()
        self.left.fit(
            X=X.loc[index_left],
            y=y.loc[index_left],
            X_true=X_true.loc[index_left_true],
            y_true=y_true.loc[index_left_true],
        )
        self.right.fit(
            X=X.loc[index_right],
            y=y.loc[index_right],
            X_true=X_true.loc[index_right_true],
            y_true=y_true.loc[index_right_true],
        )

        X_middle = X.copy()
        X_middle[self.feature] = np.nan
        self.middle.fit(
            X_middle,
            y,
            X_true=X_true.loc[index_middle_true],
            y_true=y_true.loc[index_middle_true],
        )

        self.node_importance = self._calculate_importance()

    def _find_split(self, X, y) -> tuple:
        """Calculate the best split for a decision tree

        Args:
            X: Covariates to choose from
            y: response to fit nodes to

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
                loss = self._calculate_split_loss(X, y, feature, splitter)
                if loss < loss_best:
                    loss_best = loss
                    best_feature, best_splitter = (
                        feature,
                        splitter,
                    )

        return best_feature, best_splitter

    def _calculate_split_loss(self, X, y, feature, splitter):
        """Calculates the sum of squared errors for this split

        Args:
            X: covariate vector
            y: response vector
            feature: feature of X to split test_data on
            splitter: threshold or set of categories that will go to the left node

        Returns:
            Total loss of this split for all daughter nodes
        """
        index_left, index_right = get_indices(X[feature], splitter)
        index_middle = (~index_left) & (~index_right)

        # To avoid hyperparameter-illegal splits
        if (sum(index_left) < self.min_samples_leaf) or (
            sum(index_right) < self.min_samples_leaf
        ):
            return self.loss

        loss_left_weighted = calculate_loss(y=y.loc[index_left]) * sum(index_left)
        loss_right_weighted = calculate_loss(y=y.loc[index_right]) * sum(index_right)
        loss_middle_weighted = calculate_loss(
            y=y.loc[index_middle], y_hat=self.y_hat, y_prob=self.y_prob
        ) * sum(index_middle)
        return (
            loss_left_weighted + loss_right_weighted + loss_middle_weighted
        ) / self.n

    def _initiate_daughter_nodes(self):
        """Create daughter nodes

        Return:
            tuple of three Trees. The one in the middle is None for non-trinary trees.
        """
        left = TrinaryTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            categories=self.categories,
        )
        middle = TrinaryTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth,
            categories=self.categories,
        )
        right = TrinaryTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            categories=self.categories,
        )
        return left, middle, right

    def _calculate_importance(self):
        """ "Calculate node importance for the split in this node

        Return:
            Node importance as a float
        """
        # If no values of the training test_data actually end up here it is of no importance
        if self.n_true == 0:
            importance = 0
        else:
            importance = (
                self.loss_true
                - (
                    self.left.n_true * self.left.loss_true
                    + self.middle.n_true * self.middle.loss_true
                    + self.right.n_true * self.right.loss_true
                )
                / self.n_true
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
            node_importances = self.middle._get_node_importances(
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
                index_middle = (~index_left) & (~index_right)
                y_prob.loc[index_left] = self.left.predict(X.loc[index_left], prob=True)
                y_prob.loc[index_right] = self.right.predict(
                    X.loc[index_right], prob=True
                )
                y_prob.loc[index_middle] = self.middle.predict(
                    X.loc[index_middle], prob=True
                )

        else:
            y_hat = pd.Series(index=X.index, dtype=self.response_type)
            if self.left is None:
                y_hat.loc[:] = self.y_hat
            else:
                index_left, index_right = get_indices(X[self.feature], self.splitter)
                index_middle = (~index_left) & (~index_right)

                y_hat.loc[index_left] = self.left.predict(X.loc[index_left])
                y_hat.loc[index_right] = self.right.predict(X.loc[index_right])
                y_hat.loc[index_middle] = self.middle.predict(X.loc[index_middle])

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
        print(hspace + f"Number of observations: {self.n_true}")
        if isinstance(self.y_hat, float):
            print(hspace + f"Response estimate: {np.round(self.y_hat,2)}")
        else:
            print(hspace + f"Response estimate: {self.y_hat}")
        print(hspace + f"loss: {np.round(self.loss_true,2)}")
        if self.left is not None:
            if self.feature_type == "float":
                left_rule = f"if {self.feature} <  {np.round(self.splitter,2)}"
                right_rule = f"if {self.feature} >=  {np.round(self.splitter,2)}"
            elif self.feature_type == "object":
                left_rule = f"if {self.feature} is " + ", ".join(self.splitter["left"])
                right_rule = f"if {self.feature} is " + ", ".join(
                    self.splitter["right"]
                )
            middle_rule = f"if {self.feature} n/a"

            print(hspace + f"{left_rule}:")
            self.left.print()
            print(hspace + f"{middle_rule}:")
            self.middle.print()
            print(hspace + f"{right_rule}:")
            self.right.print()


if __name__ == "__main__":
    """Main function to make the file run- and debuggable."""
    folder_path = (
        "/home/heza7322/PycharmProjects/missing-value-handling-in-carts/tests/test_data"
    )

    df_train = pd.read_csv(
        f"{folder_path}/train_data_trinary.csv",
        index_col=0,
    )

    missing_prob = 0.0
    for feature in ["X_0", "X_1"]:
        to_remove = np.random.binomial(1, missing_prob, len(df_train))
        df_train.loc[to_remove, feature] = np.nan

    df_test = pd.read_csv(
        f"{folder_path}/test_data_trinary.csv",
        index_col=0,
    )

    tree = TrinaryTree(max_depth=2)
    tree.fit(df_train[["X_0", "X_1"]], df_train["y"])

    df_test["y_hat"] = tree.predict(df_test[["X_0", "X_1"]])

    print(get_feature_importance(tree))
