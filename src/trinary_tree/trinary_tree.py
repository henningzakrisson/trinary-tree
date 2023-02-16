import itertools
import warnings
import pandas as pd
import numpy as np
import copy
from src.exceptions_and_warnings.custom_exceptions import MissingValuesInRespnonse, CantPrintUnfittedTree
from src.exceptions_and_warnings.custom_warnings import MissingFeatureWarning, ExtraFeatureWarning


class TrinaryRegressionTree:
    """
    Class to grow trinary regression trees for missing data
    """
    def __init__(
        self,
        min_samples_split=20,
        max_depth=2,
        depth=0,
        node_index = 0
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.node_index = node_index

        self.n = 0
        self.yhat = None
        self.sse = None
        self.feature = None
        self.available_features = []
        self.threshold = None
        self.left = None
        self.middle = None
        self.right = None
        self.node_importance = 0

    def fit(self, X, y, X_true = None, y_true = None):
        """Recursive method to fit the decision tree

        X_true, y_true corresponds to training data ending up in this node
        X, y corresponds to training data
        They are equal for all non-middle nodes
        """
        X = X.values if isinstance(X,pd.DataFrame) else X
        y = y.values if isinstance(y,pd.Series) else y

        # If true dataset not provided, training set is true dataset
        if X_true is None:
            X_true = X
            y_true = y
        else:
            X_true = X_true.values if isinstance(X_true,pd.DataFrame) else X_true
            y_true = y_true.values if isinstance(y_true,pd.Series) else y_true

        if np.any(np.isnan(y)) or np.any(np.isnan(y_true)):
            raise MissingValuesInRespnonse("n/a not allowed in response (y)")

        self.available_features = np.arange(X.shape[1])
        self.yhat = y.mean()
        self.sse = ((y-self.yhat)**2).sum()
        self.sse_true = ((y_true-self.yhat)**2).sum()
        self.n = len(y)
        self.n_true = len(y_true)

        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_split):
            return

        self.feature, self.threshold = self._find_split(X, y)
        if self.feature is None:
            return

        index_left = X[:, self.feature] < self.threshold
        index_right = X[:, self.feature] >= self.threshold

        self.left, self.middle, self.right = self._initiate_daughter_nodes()
        self.left.fit(X[index_left], y[index_left])
        self.right.fit(X[index_right], y[index_right])

        X_middle = X.copy()
        X_middle[:,self.feature] = np.nan
        if (len(X_true)==0) or (len(y_true)==0):
            self.middle.fit(X_middle, y, X_true = X_true, y_true = y_true)
        else:
            index_middle_true = np.isnan(X_true[:,self.feature])
            self.middle.fit(X_middle, y, X_true = X_true[index_middle_true], y_true = y_true[index_middle_true])

        self.node_importance = self._calculate_importance()

    def _find_split(self, X, y) -> tuple:
        """"Calculate the best split for a decision tree"""
        sse_best = self.sse
        best_feature, best_threshold = None, None
        split_candidates = self._get_split_candidates(X)

        for feature, threshold in split_candidates:
            sse = self._calculate_split_sse(X,y,feature,threshold)
            if sse < sse_best:
                sse_best = sse
                best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _get_split_candidates(self,X):
        features = [feature for feature in range(X.shape[1]) if sum(np.isnan(X[:,feature]))<len(X)]
        thresholds = [self._get_threshold_candidates(X[:,feature]) for feature in range(X.shape[1])]
        combinations = [list(itertools.product([feature],thresholds[feature])) for feature in features]
        return list(itertools.chain.from_iterable(combinations))

    def _get_threshold_candidates(self,X):
        if np.all(np.isnan(X)):
            return []
        values = np.sort(np.unique(X[~np.isnan(X)]))
        numbers_between_values = np.convolve(values, np.ones(2), 'valid') / 2
        return numbers_between_values

    def _calculate_split_sse(self,X,y,feature,threshold):
        index_left = (X[:, feature] < threshold)
        index_middle = np.isnan(X[:,feature])
        index_right = X[:, feature] >= threshold

        # To avoid hyperparameter-illegal splits
        if np.any([(sum(index)<self.min_samples_split) for index in [index_left,index_right]]):
            return self.sse

        sse_left = ((y[index_left] - y[index_left].mean())**2).sum()
        if sum(index_middle)>0:
            sse_middle = ((y[index_middle] - y[index_middle].mean())**2).sum()
        else:
            sse_middle = 0
        sse_right = ((y[index_right] - y[index_right].mean())**2).sum()
        return sse_left + sse_middle + sse_right

    def _initiate_daughter_nodes(self):
        left = TrinaryRegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    node_index = 3*self.node_index
                    )
        middle = TrinaryRegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    node_index = 3*self.node_index + 1
                    )
        right = TrinaryRegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    node_index = 3*self.node_index + 2
                    )
        return left, middle, right

    def _calculate_importance(self):
        if self.n_true ==0:
            return 0
        else:
            return self.sse_true - (self.left.n_true*self.left.sse_true + self.right.n_true*self.right.sse_true)/self.n_true

    def feature_importance(self):
        node_importances = self._get_node_importances(node_importances = {feature: [] for feature in self.available_features})
        total_importances = {feature: sum(node_importances[feature]) for feature in node_importances}
        feature_importances = {feature: total_importances[feature]/sum(total_importances.values()) for feature in total_importances}
        return feature_importances

    def _get_node_importances(self, node_importances):
        if self.feature is not None:
            node_importances[self.feature].append(self.node_importance)
        if self.left is not None:
            node_importances = self.left._get_node_importances(node_importances = node_importances)
        if self.right is not None:
            node_importances = self.right._get_node_importances(node_importances = node_importances)

        return node_importances


    def predict(self,X):
        """Recursive method to predict from new of features"""
        X = X.values if isinstance(X,pd.DataFrame) else X
        if X.shape[1] < len(self.available_features):
            warnings.warn('Covariate matrix missing features - filling with n/a',MissingFeatureWarning)
            X_fill = np.ones((len(X), len(self.available_features) - X.shape[1]))
            X = np.c_[X, X_fill]
        elif X.shape[1] > len(self.available_features):
            warnings.warn('Covariate matrix contains redundant features',ExtraFeatureWarning)

        y_hat = np.ones(len(X))*np.nan

        if self.left is None:
            y_hat[:] = self.yhat
            return y_hat

        index_left = X[:, self.feature] < self.threshold
        index_middle = np.isnan(X[:,self.feature])
        index_right = X[:, self.feature] >= self.threshold

        y_hat[index_left] = self.left.predict(X[index_left])
        y_hat[index_middle] = self.middle.predict(X[index_middle])
        y_hat[index_right] = self.right.predict(X[index_right])

        return y_hat

    def print(self):
        """ Print the tree structure"""
        if self.yhat is None:
            raise CantPrintUnfittedTree("Can't print tree before fitting to data")

        hspace = '---'*self.depth
        print(hspace+f'Number of observations: {self.n_true}')
        print(hspace+f'Response estimate: {np.round(self.yhat,2)}')
        print(hspace+f'SSE: {np.round(self.sse_true,2)}')
        if self.left is not None:
            left_rule  = f'if {self.feature} <  {np.round(self.threshold,2)}'
            middle_rule = f'if {self.feature} n/a'
            right_rule = f'if {self.feature} >=  {np.round(self.threshold,2)}'

            print(hspace+f'{left_rule}:')
            self.left.print()
            print(hspace+f'{middle_rule}:')
            self.middle.print()
            print(hspace+f'{right_rule}:')
            self.right.print()

if __name__ == '__main__':
    n = 1000
    x0 = np.arange(0,n)
    x1 = np.tile(np.arange(n/10),10)
    X = np.stack([x0,x1]).T
    y = 10 * (x0>=n/2) + 2 * (x1>=(n/10)/2)

    mask = np.zeros(X.shape,dtype='bool')
    mask[:100] = True
    np.random.shuffle(mask)
    mask = mask.reshape(X.shape)
    X[mask] = np.nan

    tree = TrinaryRegressionTree(max_depth=2, min_samples_split = 1)
    tree.fit(X = X, y = y)

    tree.print()

    print(tree.feature_importance())
