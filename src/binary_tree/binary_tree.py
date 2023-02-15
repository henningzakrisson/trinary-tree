import itertools
import pandas as pd
import numpy as np
import copy

from src.exceptions import MissingValuesInRespnonse,CantPrintUnfittedTree

class BinaryRegressionTree:
    """
    Class to grow a binary regression decision tree
    """
    def __init__(
        self,
        min_samples_split=20,
        max_depth=2,
        depth=0,
        node_index = 0,
        missing_rule = 'majority'
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.node_index = node_index
        self.missing_rule = missing_rule

        self.n = 0
        self.yhat = None
        self.sse = None
        self.feature = None
        self.available_features = []
        self.threshold = None
        self.default_split = None
        self.left = None
        self.right = None
        self.node_importance = 0

    def fit(self, X, y):
        """Recursive method to fit the decision tree"""
        if np.any(np.isnan(y)):
            raise MissingValuesInRespnonse("n/a not allowed in response (y)")

        X = X.values if isinstance(X,pd.DataFrame) else X
        y = y.values if isinstance(y,pd.Series) else y

        self.available_features = np.arange(X.shape[1])
        self.yhat = y.mean()
        self.sse = ((y-self.yhat)**2).sum()
        self.n = len(y)

        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_split):
            return

        self.feature, self.threshold, self.default_split = self._find_split(X, y)
        if self.feature is None:
            return

        index_left = (X[:, self.feature] < self.threshold)
        if self.default_split == 'left':
            index_left |= np.isnan(X[:, self.feature])

        self.left, self.right = self._initiate_daughter_nodes()
        self.left.fit(X[index_left], y[index_left])
        self.right.fit(X[~index_left], y[~index_left])

        self.node_importance = self._calculate_importance()

    def _find_split(self, X, y) -> tuple:
        """Calculate the best split for a decision tree"""
        sse_best = self.sse
        best_feature, best_threshold, best_default_split = None, None, None
        split_candidates = self._get_split_candidates(X)

        for feature, threshold, default_split in split_candidates:
            sse = self._calculate_split_sse(X,y,feature,threshold,default_split)
            if sse < sse_best:
                sse_best = sse
                best_feature, best_threshold, best_default_split = feature, threshold, default_split

        return best_feature, best_threshold, best_default_split

    def _get_split_candidates(self,X):
        features = [feature for feature in range(X.shape[1]) if sum(np.isnan(X[:,feature]))<len(X)]
        thresholds = [self._get_threshold_candidates(X[:,feature]) for feature in features]

        if self.missing_rule == 'mia':
            default_splits = ['left','right']
            combinations = [list(itertools.product([features[feature]],thresholds[feature],default_splits)) for feature in features]
            return list(itertools.chain.from_iterable(combinations))

        elif self.missing_rule == 'majority':
            feature_threshold_combinations = [list(itertools.product([features[feature]],thresholds[feature])) for feature in features]
            feature_threshold_candidates = list(itertools.chain.from_iterable(feature_threshold_combinations))
            default_splits = ['left' if sum(X[:,feature]<threshold)>sum(X[:,feature]>=threshold) else 'right' for feature,threshold in feature_threshold_candidates]
            return [feature_threshold +(default_rule,) for feature_threshold,default_rule in zip(feature_threshold_candidates,default_splits)]

    def _get_threshold_candidates(self,X):
        values = np.sort(np.unique(X[~np.isnan(X)]))
        numbers_between_values = np.convolve(values, np.ones(2), 'valid') / 2
        return numbers_between_values

    def _calculate_split_sse(self,X,y,feature,threshold,default_split):
        index_left = (X[:, feature] < threshold)
        if default_split == 'left':
            index_left |= np.isnan(X[:, feature])

        # To avoid hyperparameter-illegal splits
        if (sum(index_left)<self.min_samples_split) or (sum(~index_left)<self.min_samples_split):
            return self.sse

        sse_left = ((y[index_left] - y[index_left].mean())**2).sum()
        sse_right = ((y[~index_left] - y[~index_left].mean())**2).sum()
        return sse_left + sse_right

    def _initiate_daughter_nodes(self):
        left = BinaryRegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    node_index = 2*self.node_index,
                    missing_rule= self.missing_rule
                    )
        right = BinaryRegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    node_index = 2*self.node_index + 1,
                    missing_rule= self.missing_rule
                    )
        return left, right

    def _calculate_importance(self):
        return self.sse - (self.left.n*self.left.sse + self.right.n*self.right.sse)/self.n

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
        y_hat = np.ones(len(X))*np.nan

        if self.left is None:
            y_hat[:] = self.yhat
            return y_hat

        index_left = (X[:, self.feature] < self.threshold)
        if self.default_split == 'left':
            index_left |= np.isnan(X[:, self.feature])

        y_hat[index_left] = self.left.predict(X[index_left])
        y_hat[~index_left] = self.right.predict(X[~index_left])

        return y_hat

    def print(self):
        """ Print the tree structure"""
        if self.yhat is None:
            raise CantPrintUnfittedTree("Can't print tree before fitting to data")

        hspace = '---'*self.depth
        print(hspace+f'Number of observations: {self.n}')
        print(hspace+f'Response estimate: {np.round(self.yhat,2)}')
        if self.left is not None:
            left_rule  = f'if {self.feature} <  {np.round(self.threshold,2)}'
            right_rule = f'if {self.feature} >=  {np.round(self.threshold,2)}'
            if self.default_split=='left':
                left_rule += ' or n/a'
            else:
                right_rule += ' or n/a'
            print(hspace+f'{left_rule}:')
            self.left.print()

            print(hspace+f'{right_rule}:')
            self.right.print()

if __name__ == '__main__':
    n = 1000
    x0 = np.arange(0,n)
    x1 = np.tile(np.arange(n/10),10)
    X = np.stack([x0,x1]).T
    y = 10 * (x0>=n/2) + 2 * (x1>=(n/10)/2)

    tree = BinaryRegressionTree(max_depth=5, min_samples_split = 1)
    tree.fit(X = X, y = y)

    tree.print()
