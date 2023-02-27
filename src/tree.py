import itertools
import warnings
import pandas as pd
import numpy as np
from src.exceptions_and_warnings.custom_exceptions import (
    MissingValuesInResponse,
    CantPrintUnfittedTree,
)
from src.exceptions_and_warnings.custom_warnings import (
    MissingFeatureWarning,
    ExtraFeatureWarning,
)


class Tree:
    """Module for classification and regression trees with three different ways of handling missing data

    The missing data strategies are:
     - Majority rule: missing datapoints go to the node with the most training data
     - Missing Incorporated in Attributes (MIA): missing datapoints go to the node
      which improved the loss the most in the training data
     - Trinary split: Creates a third node for missing values, which inherits data and output from mother node
    """

    def __init__(
        self,
        min_samples_leaf=20,
        max_depth=2,
        depth=0,
        missing_rule="majority",
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
        self.missing_rule = missing_rule
        self.categories = categories

        self.n = 0
        self.n_true = 0
        self.y_hat = None
        self.y_prob = {}
        self.loss = None
        self.loss_true = None
        self.feature = None
        self.feature_type = None
        self.available_features = []
        self.threshold = None  # For continous features
        self.sets = None  # Dict with left/right keys for categorical features
        self.default_split = None
        self.left = None
        self.middle = None
        self.right = None
        self.node_importance = 0

    def fit(self, X, y, X_true=None, y_true=None):
        """Recursive method to fit the decision tree.

        Will call itself to create daughter nodes if applicable.

        Args:
            X: covariate vector (n x p). numpy array or pandas DataFrame.
            y: response vector (n x 1). numpy array or pandas Series.
            X_true: covariate vector (n x p) of data that ends up in this node
            during training. Only applicable for trinary trees - for others, X_true = X
            y_true: response vector (n x 1) of data that ends up in this node
            during training. Only applicable for trinary trees - for others, y_true = y

        Raises:
            MissingValuesInResponse: Can not fit to missing categories, thus errors out
        """
        X, y = self._fix_datatypes(X, y)

        # If true dataset not provided, training set is true dataset
        if X_true is None:
            X_true = X
            y_true = y

        if np.any(y.isna()) or np.any(y.isna()):
            raise MissingValuesInResponse("n/a not allowed in response (y)")

        self.available_features = X.columns  # This means all features  in the input
        self.n = len(y)
        self.n_true = len(y_true)

        if y.dtype == "float":
            self.y_hat = y.mean()
        else:
            if self.categories is None:
                self.categories = list(y.unique())
            self.y_prob = (y.value_counts() / len(y)).to_dict()
            self.y_hat = max(self.y_prob, key=self.y_prob.get)
            for category in self.categories:
                if category not in self.y_prob:
                    self.y_prob[category] = 0
        self.loss = self._calculate_loss(y, self.y_hat)
        self.loss_true = self._calculate_loss(y_true, self.y_hat)

        # Check pruning conditions
        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_leaf):
            return

        # Find splitting parameters
        self.feature, splitter, self.default_split = self._find_split(X, y)

        if self.feature is None:
            return
        elif X[self.feature].dtype == "float":
            self.feature_type = "float"
            self.threshold = splitter
            index_left = X[self.feature] < self.threshold
            index_right = X[self.feature] >= self.threshold
        elif X[self.feature].dtype == "object":
            self.feature_type = "object"
            self.sets = splitter
            index_left = X[self.feature].isin(self.sets["left"])
            index_right = X[self.feature].isin(self.sets["right"])
        if self.default_split == "left":
            index_left |= X[self.feature].isna()
        elif self.default_split == "right":
            index_right |= X[self.feature].isna()

        # Send data to daughter nodes
        self.left, self.middle, self.right = self._initiate_daughter_nodes()
        self.left.fit(X.loc[index_left], y.loc[index_left])
        self.right.fit(X.loc[index_right], y.loc[index_right])

        # For the trinary strategy
        if self.middle is not None:
            X_middle = X.copy()
            X_middle[self.feature] = np.nan
            index_middle_true = X_true[self.feature].isna()
            self.middle.fit(
                X_middle,
                y,
                X_true=X_true.loc[index_middle_true],
                y_true=y_true.loc[index_middle_true],
            )

        self.node_importance = self._calculate_importance()

    def _fix_datatypes(self, X, y):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        for feature in X:
            if X[feature].dtype == "int":
                X[feature] = X[feature].astype(float)
        y = pd.Series(y) if isinstance(y, np.ndarray) else y
        if y.dtype in ["float", "int"]:
            y = y.astype(float)

        return X, y

    def _find_split(self, X, y) -> tuple:
        """Calculate the best split for a decision tree

        Args:
            X: Covariates to choose from
            y: response to fit nodes to

        Returns:
            best_feature: feature to split by for minimum loss
            best_splitter: threshold or left-category-set to split feature by for minimum loss
            best_default split: node to send missing values to
        """
        # Initiate here in order to not grow more if this loss is not beaten
        loss_best = self.loss
        best_feature, best_splitter, best_default_split = None, None, None

        features = [
            feature for feature in X.columns if X[feature].isna().sum() < len(X)
        ]
        for feature in features:
            splitters = self._get_splitter_candidates(X[feature])
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

    def _get_splitter_candidates(self, x):
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

    def _get_default_split_candidates(self, X, feature, splitter):
        """Get default split candidates given the rule, covariates and features

        Args:
            X: covariate vector
            feature: feature to split on
            splitter: threshold or sets

        Return:
            list of 'left' or 'right' depending on which node gets the most data
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
        elif self.missing_rule == "mia":
            return ["left", "right"]
        elif self.missing_rule == "trinary":
            return ["middle"]

    def _calculate_split_loss(self, X, y, feature, splitter, default_split):
        """Calculates the sum of squared errors for this split

        Args:
            X: covariate vector
            y: response vector
            feature: feature of X to split data on
            splitter: threshold or set of categories that will go to the left node
            default_split: node to put missing values in

        Returns:
            Total loss of this split for all daughter nodes
        """
        if X[feature].dtype == "float":
            index_left = X[feature] < splitter
            index_right = X[feature] >= splitter
        elif X[feature].dtype == "object":
            index_left = X[feature].isin(splitter["left"])
            index_right = X[feature].isin(splitter["right"])
        if default_split == "left":
            index_left |= X[feature].isna()
        elif default_split == "right":
            index_right |= X[feature].isna()
        elif default_split == "middle":
            index_middle = X[feature].isna()

        # To avoid hyperparameter-illegal splits
        if (sum(index_left) < self.min_samples_leaf) or (
            sum(index_right) < self.min_samples_leaf
        ):
            return self.loss

        loss_left_weighted = self._calculate_loss(y=y.loc[index_left]) * sum(index_left)
        loss_right_weighted = self._calculate_loss(y=y.loc[index_right]) * sum(
            index_right
        )
        if default_split == "middle":
            loss_middle_weighted = self._calculate_loss(y=y.loc[index_middle]) * sum(
                index_right
            )
        else:
            loss_middle_weighted = 0

        return (
            loss_left_weighted + loss_right_weighted + loss_middle_weighted
        ) / self.n

    def _calculate_loss(self, y, y_hat=None):
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

    def _initiate_daughter_nodes(self):
        """Create daughter nodes

        Return:
            tuple of three Trees. The one in the middle is None for non-trinary trees.
        """
        left = Tree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            missing_rule=self.missing_rule,
            categories=self.categories,
        )
        right = Tree(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            depth=self.depth + 1,
            missing_rule=self.missing_rule,
            categories=self.categories,
        )

        if self.missing_rule == "trinary":
            middle = Tree(
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                depth=self.depth + 1,
                missing_rule=self.missing_rule,
                categories=self.categories,
            )
        else:
            middle = None
        return left, middle, right

    def _calculate_importance(self):
        """ "Calculate node importance for the split in this node

        Return:
            Node importance as a float
        """
        # If no values of the training data actually end up here it is of no importance
        if self.n_true == 0:
            return 0
        elif self.default_split == "trinary":
            return (
                self.loss_true
                - (
                    self.left.n_true * self.left.loss_true
                    + self.middle.n_true * self.middle.loss_true
                    + self.right.n_true * self.right.loss_true
                )
                / self.n_true
            )
        else:
            return (
                self.loss_true
                - (
                    self.left.n_true * self.left.loss_true
                    + self.right.n_true * self.right.loss_true
                )
                / self.n_true
            )

    def feature_importance(self):
        """Calculate feature importance for all features in X

        Return:
            dict with keys corresponding to feature and values corresponding to their feature importances. Sums to 1.
        """
        node_importances = self._get_node_importances(
            node_importances={feature: [] for feature in self.available_features}
        )
        total_importances = {
            feature: sum(node_importances[feature]) for feature in node_importances
        }
        feature_importances = {
            feature: total_importances[feature] / sum(total_importances.values())
            for feature in total_importances
        }
        return feature_importances

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
        if self.right is not None:
            node_importances = self.right._get_node_importances(
                node_importances=node_importances
            )
        if self.middle is not None:
            node_importances = self.middle._get_node_importances(
                node_importances=node_importances
            )
        return node_importances

    def predict(self, X, prob=False):
        """Recursive method to predict from new of features

        Args:
            Covariate vector X (m x p) of same secondary dimension as training covariate vector
            prob: True if predict probabilities rather than

        Returns:
            response predictions y_hat as a pandas Series. DataFrame if probabilities.
        """
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X
        missing_features = [
            feature for feature in self.available_features if feature not in X.columns
        ]
        if len(missing_features) > 0:
            warnings.warn(
                f"Covariate matrix missing features {missing_features} - filling with n/a",
                MissingFeatureWarning,
            )
            for feature in missing_features:
                X[feature] = np.nan

        extra_features = [
            feature for feature in X.columns if feature not in self.available_features
        ]
        if len(extra_features) > 0:
            warnings.warn(
                f"Covariate matrix missing features {extra_features} - filling with n/a",
                ExtraFeatureWarning,
            )
            X = X.drop(extra_features, axis=1)

        if prob:
            y_hat = pd.DataFrame(index=X.index, columns=self.categories, dtype=float)
        elif self.categories is None:
            y_hat = pd.Series(index=X.index, dtype=float)
        else:
            y_hat = pd.Series(index=X.index, dtype=object)

        if self.left is None:
            if not prob:
                y_hat.loc[:] = self.y_hat
            else:
                for category in self.categories:
                    y_hat[category] = self.y_prob[category]
            return y_hat

        if self.feature_type == "float":
            index_left = X[self.feature] < self.threshold
            index_right = X[self.feature] >= self.threshold
        elif self.feature_type == "object":
            index_left = X[self.feature].isin(self.sets["left"])
            index_right = X[self.feature].isin(self.sets["right"])
        if self.default_split == "left":
            index_left |= X[self.feature].isna()
        elif self.default_split == "right":
            index_right |= X[self.feature].isna()
        elif self.default_split == "middle":
            index_middle = X[self.feature].isna()
            y_hat.loc[index_middle] = self.middle.predict(
                X.loc[index_middle], prob=prob
            )

        y_hat.loc[index_left] = self.left.predict(X.loc[index_left], prob=prob)
        y_hat.loc[index_right] = self.right.predict(X.loc[index_right], prob=prob)

        return y_hat

    def print(self):
        """Print the tree structure"""
        if self.y_hat is None:
            raise CantPrintUnfittedTree("Can't print tree before fitting to data")

        hspace = "---" * self.depth
        print(hspace + f"Number of observations: {self.n_true}")
        if isinstance(self.y_hat, float):
            print(hspace + f"Response estimate: {np.round(self.y_hat,2)}")
        else:
            print(hspace + f"Response estimate: {self.y_hat}")
        print(hspace + f"loss: {np.round(self.loss_true,2)}")
        if self.left is not None:
            if self.feature_type == "float":
                left_rule = f"if {self.feature} <  {np.round(self.threshold,2)}"
                right_rule = f"if {self.feature} >=  {np.round(self.threshold,2)}"
            elif self.feature_type == "object":
                left_rule = f"if {self.feature} is " + ", ".join(self.sets["left"])
                right_rule = f"if {self.feature} is " + ", ".join(self.sets["right"])
            if self.default_split == "middle":
                middle_rule = f"if {self.feature} n/a"
            elif self.default_split == "left":
                left_rule += " or n/a"
            else:
                right_rule += " or n/a"

            print(hspace + f"{left_rule}:")
            self.left.print()
            if self.default_split == "middle":
                print(hspace + f"{middle_rule}:")
                self.middle.print()
            print(hspace + f"{right_rule}:")
            self.right.print()


if __name__ == "__main__":
    """Main function to make the file run- and debuggable."""
    df = pd.read_csv(
        "/home/heza7322/PycharmProjects/missing-value-handling-in-carts/tests/test_tree/data/test_data_class.csv",
        index_col=0,
    )
    X = df.drop("y", axis=1)
    y = df["y"]

    tree = Tree(max_depth=3, min_samples_leaf=20)
    tree.fit(X, y)

    df["y_hat"] = tree.predict(X)
    df_probs = tree.predict(X, prob=True)
