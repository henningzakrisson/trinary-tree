import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Dict
from src.trees.utils import (
    CantPrintUnfittedTree,
    fix_datatypes,
    fit_response,
    calculate_loss,
    check_terminal_node,
    get_splitter_candidates,
    get_indices,
    check_features,
)


class BinaryTree:
    """Module for classification and regression trees with standard handling missing test_data

    The missing test_data strategies are:
     - Majority rule: missing datapoints go to the node with the most training test_data
     - Missing Incorporated in Attributes (MIA): missing datapoints go to the node
      which improved the loss the most in the training test_data
    """

    def __init__(
        self,
        min_samples_leaf: int = 20,
        max_depth: int = 2,
        depth: int = 0,
        categories: Union[None, List[str]] = None,
        missing_rule: str = "majority",
    ):
        """Initialize the tree. Will be called recursively to create daughter nodes.

        :param min_samples_leaf: Minimum number of samples in a leaf node
        :param max_depth: Maximum depth of the tree
        :param depth: Current depth of the tree
        :param categories: Categories of the response variable. None if regression.
        :param missing_rule: Missing test_data strategy. Either "majority" or "mia"
        """
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.depth = depth
        self.categories = categories
        self.missing_rule = missing_rule
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

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Recursive method to fit the decision tree. Will call itself to create daughter nodes if applicable.

        :param X: Training features
        :param y: Training response
        """
        X, y, _ = fix_datatypes(X, y)
        self.features = X.columns
        self.n = len(y)

        self.y_hat, self.y_prob, self.categories = fit_response(y, self.categories)
        if self.categories is not None:
            self.response_type = "object"
        else:
            self.response_type = "float"

        self.loss = calculate_loss(y)

        # Check pruning conditions
        if check_terminal_node(self):
            self.left, self.right = None, None
            return

        # Find splitting parameters
        self.feature, self.splitter, self.default_split = self._find_split(X, y)

        if self.feature is None:
            self.left, self.right = None, None
            return

        self.feature_type = "float" if X[self.feature].dtype == "float" else "object"

        index_left, index_right = get_indices(
            X[self.feature], self.splitter, self.default_split
        )

        # Send test_data to daughter nodes
        self.left, self.right = self._initiate_daughter_nodes()
        self.left.fit(X.loc[index_left], y.loc[index_left])
        self.right.fit(X.loc[index_right], y.loc[index_right])

        self.node_importance = self._calculate_importance()

    def _find_split(self, X: pd.DataFrame, y: pd.Series):
        """Calculate the best split for a decision tree.

        :param X: Training features
        :param y: Training response

        :return: Best feature, best splitter, best default split
        """
        # Initiate here in order to not grow more if this loss is not beaten
        loss_best = self.loss
        best_feature, best_splitter, best_default_split = None, None, None

        features = [
            feature for feature in X.columns if X[feature].isna().sum() < len(X)
        ]
        for feature in features:
            splitters = get_splitter_candidates(X[feature])
            for splitter in splitters:
                default_splits = self._get_default_split_candidates(
                    X, feature, splitter
                )
                for default_split in default_splits:
                    loss = self._calculate_split_loss(
                        X, y, feature, splitter, default_split
                    )
                    if loss < loss_best:
                        loss_best = loss
                        best_feature, best_splitter, best_default_split = (
                            feature,
                            splitter,
                            default_split,
                        )

        return best_feature, best_splitter, best_default_split

    def _get_default_split_candidates(
        self, X, feature, splitter
    ):
        """Get default split candidates given the rule, covariates and features

        :param X: Training features
        :param feature: Feature to split on
        :param splitter: Splitter to split with
        """
        if (self.missing_rule == "majority") or (
            (self.missing_rule == "mia") & (X[feature].isna().sum() == 0)
        ):
            if isinstance(splitter, float):
                return (
                    ["left"]
                    if sum(X[feature] < splitter) > sum(X[feature] >= splitter)
                    else ["right"]
                )
            elif isinstance(splitter, dict):
                return (
                    ["left"]
                    if sum(X[feature].isin(splitter["left"]))
                    > sum(X[feature].isin(splitter["right"]))
                    else ["right"]
                )
        else:  # mia strategy
            return ["left", "right"]

    def _calculate_split_loss(self, X: pd.DataFrame,
                              y: pd.Series,
                              feature: str,
                              splitter: Union[float, Dict[str: str]],
                              default_split: str) -> float:
        """Calculates the sum of squared errors for this split

        :param X: Training features
        :param y: Training response
        :param feature: Feature to split on
        :param splitter: Splitter to split with
        :param default_split: Default split to use if missing_rule is mia

        :return: loss for this split
        """
        index_left, index_right = get_indices(X[feature], splitter, default_split)

        # To avoid hyperparameter-illegal splits
        if (sum(index_left) < self.min_samples_leaf) or (
            sum(index_right) < self.min_samples_leaf
        ):
            return self.loss

        loss_left_weighted = calculate_loss(y=y.loc[index_left]) * sum(index_left)
        loss_right_weighted = calculate_loss(y=y.loc[index_right]) * sum(index_right)

        return (loss_left_weighted + loss_right_weighted) / self.n

    def _initiate_daughter_nodes(self) -> Tuple[Tree, Tree]:
        """Create daughter nodes

        :return: Tuple of left and right daughter nodes
        """
        left = BinaryTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            missing_rule=self.missing_rule,
            categories=self.categories,
        )
        right = BinaryTree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            missing_rule=self.missing_rule,
            categories=self.categories,
        )

        return left, right

    def _calculate_importance(self):
        """ "Calculate node importance for the split in this node

        :return: node importance
        """
        # If no values of the training test_data actually end up here it is of no importance
        if self.n == 0:
            return 0
        else:
            return (
                self.loss
                - (self.left.n * self.left.loss + self.right.n * self.right.loss)
                / self.n
            )

    def _get_node_importances(self, node_importances: Dict[str, List]):
        """Get node importances for this node and all its daughters

        :param node_importances: Dictionary of previous node importances

        :return: Updated dictionary of node importances
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

    def predict(self, X: Union[np.ndarray, pd.DataFrame],
                prob: bool=False):
        """Recursive method to predict from new of features

        :param X: Features to predict from
        :param prob: Whether to return category probabilities or response estimates

        :return: Predictions
        """
        X, _, _ = fix_datatypes(X)
        X = check_features(X, self.features)

        if self.response_type == "object":
            y_prob = pd.DataFrame(index=X.index, columns=self.categories, dtype=float)
            if self.left is None:
                for category in self.categories:
                    y_prob[category] = self.y_prob[category]
            else:
                index_left, index_right = get_indices(
                    X[self.feature], self.splitter, default_split=self.default_split
                )

                y_prob.loc[index_left] = self.left.predict(X.loc[index_left], prob=True)
                y_prob.loc[index_right] = self.right.predict(
                    X.loc[index_right], prob=True
                )

        else:
            y_hat = pd.Series(index=X.index, dtype=self.response_type)
            if self.left is None:
                y_hat.loc[:] = self.y_hat
            else:
                index_left, index_right = get_indices(
                    X[self.feature], self.splitter, default_split=self.default_split
                )

                y_hat.loc[index_left] = self.left.predict(X.loc[index_left])
                y_hat.loc[index_right] = self.right.predict(X.loc[index_right])

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
        print(hspace + f"Number of observations: {self.n}")
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
            if self.default_split == "left":
                left_rule += " or n/a"
            else:
                right_rule += " or n/a"
            print(hspace + f"{left_rule}:")
            self.left.print()
            print(hspace + f"{right_rule}:")
            self.right.print()


if __name__ == "__main__":
    """Main function to make the file run- and debuggable."""
    file_path = "/tests/test_data/test_data_cat.csv"

    df = pd.read_csv(file_path, index_col=0)
    X = df.drop("y", axis=1)
    y = df["y"]

    max_depth = 4

    tree = BinaryTree(max_depth=max_depth)
    tree.fit(X, y)
    y_hat = tree.predict(X)

    print("h")
