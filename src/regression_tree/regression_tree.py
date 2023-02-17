import itertools
import warnings
import pandas as pd
import numpy as np
from src.exceptions_and_warnings.custom_exceptions import MissingValuesInRespnonse, CantPrintUnfittedTree
from src.exceptions_and_warnings.custom_warnings import MissingFeatureWarning, ExtraFeatureWarning

class RegressionTree:
    """
    Class to grow a binary regression decision tree
    """
    def __init__(
        self,
        min_samples_split=20,
        max_depth=2,
        depth=0,
        missing_rule = 'majority',
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.missing_rule = missing_rule

        self.n = 0
        self.n_true = 0
        self.yhat = None
        self.sse = None
        self.sse_true = None
        self.feature = None
        self.available_features = []
        self.threshold = None
        self.default_split = None
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

        X = X.values if isinstance(X,pd.DataFrame) else X
        y = y.values if isinstance(y,pd.Series) else y

        self.available_features = np.arange(X.shape[1])
        self.yhat = y.mean()
        self.sse = ((y-self.yhat)**2).sum()
        self.sse_true = ((y_true-self.yhat)**2).sum()
        self.n = len(y)
        self.n_true = len(y_true)

        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_split):
            return

        self.feature, self.threshold, self.default_split = self._find_split(X, y)
        if self.feature is None:
            return

        index_left = X[:, self.feature] < self.threshold
        index_right = X[:, self.feature] >= self.threshold
        if self.default_split == 'left':
            index_left |= np.isnan(X[:, self.feature])
        elif self.default_split == 'right':
            index_right |= np.isnan(X[:, self.feature])

        self.left, self.middle, self.right = self._initiate_daughter_nodes()
        self.left.fit(X[index_left], y[index_left])
        self.right.fit(X[index_right], y[index_right])

        if self.middle is not None:
            X_middle = X.copy()
            X_middle[:,self.feature] = np.nan
            if (len(X_true)==0) or (len(y_true)==0):
                self.middle.fit(X_middle, y, X_true = X_true, y_true = y_true)
            else:
                index_middle_true = np.isnan(X_true[:,self.feature])
                self.middle.fit(X_middle,
                                y,
                                X_true = X_true[index_middle_true],
                                y_true = y_true[index_middle_true])

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
        thresholds = {feature:self._get_threshold_candidates(X[:,feature]) for feature in features}

        if self.missing_rule == 'mia':
            default_splits = ['right','left'] # This order to ensure right is default split
            combinations = [list(itertools.product([feature],thresholds[feature],default_splits)) for feature in features]
            return list(itertools.chain.from_iterable(combinations))

        elif self.missing_rule == 'majority':
            feature_threshold_combinations = [list(itertools.product([feature],thresholds[feature])) for feature in features]
            feature_threshold_candidates = list(itertools.chain.from_iterable(feature_threshold_combinations))
            default_splits = ['left' if sum(X[:,feature]<threshold)>sum(X[:,feature]>=threshold) else 'right' for feature,threshold in feature_threshold_candidates]
            return [feature_threshold +(default_rule,) for feature_threshold,default_rule in zip(feature_threshold_candidates,default_splits)]

        elif self.missing_rule == 'trinary':
            default_splits = ['middle']
            combinations = [list(itertools.product([feature],thresholds[feature],default_splits)) for feature in features]
            return list(itertools.chain.from_iterable(combinations))

    def _get_threshold_candidates(self,X):
        if np.all(np.isnan(X)):
            return []
        values = np.sort(np.unique(X[~np.isnan(X)]))
        numbers_between_values = np.convolve(values, np.ones(2), 'valid') / 2
        return numbers_between_values

    def _calculate_split_sse(self,X,y,feature,threshold,default_split):
        index_left = X[:, feature] < threshold
        index_right = X[:,feature] >= threshold
        if default_split == 'left':
            index_left |= np.isnan(X[:, feature])
        elif default_split == 'right':
            index_right |= np.isnan(X[:, feature])
        elif default_split == 'middle':
            index_middle = np.isnan(X[:, feature])

        # To avoid hyperparameter-illegal splits
        if (sum(index_left)<self.min_samples_split) or (sum(index_right)<self.min_samples_split):
            return self.sse

        sse_left = ((y[index_left] - y[index_left].mean())**2).sum()
        sse_right = ((y[index_right] - y[index_right].mean())**2).sum()
        if default_split == 'middle' and sum(index_middle)>0:
            sse_middle = ((y[index_middle] - y[index_middle].mean())**2).sum()
        else:
            sse_middle = 0
        return sse_left + sse_middle + sse_right

    def _initiate_daughter_nodes(self):
        left = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    missing_rule= self.missing_rule,
                    )
        right = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    missing_rule= self.missing_rule,
                    )

        if self.missing_rule == 'trinary':
            middle = RegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    missing_rule= self.missing_rule,
                    )
        else:
            middle = None
        return left, middle, right

    def _calculate_importance(self):
        if self.n_true ==0:
            return 0
        elif self.default_split == 'trinary':
            return self.sse_true - (self.left.n_true*self.left.sse_true + self.middle.n_true*self.middle.sse_true + self.right.n_true*self.right.sse_true)/self.n_true
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
        if self.middle is not None:
            node_importances = self.middle._get_node_importances(node_importances = node_importances)
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
        index_right = X[:, self.feature] >= self.threshold
        if self.default_split == 'left':
            index_left |= np.isnan(X[:, self.feature])
        elif self.default_split == 'right':
            index_right |= np.isnan(X[:, self.feature])
        elif self.default_split == 'middle':
            index_middle = np.isnan(X[:,self.feature])
            y_hat[index_middle] = self.middle.predict(X[index_middle])

        y_hat[index_left] = self.left.predict(X[index_left])
        y_hat[index_right] = self.right.predict(X[index_right])

        return y_hat

    def print(self):
        """ Print the tree structure"""
        if self.yhat is None:
            raise CantPrintUnfittedTree("Can't print tree before fitting to data")

        hspace = '---'*self.depth
        print(hspace+f'Number of observations: {self.n}')
        print(hspace+f'Response estimate: {np.round(self.yhat,2)}')
        print(hspace+f'SSE: {np.round(self.sse,2)}')
        if self.left is not None:
            left_rule  = f'if {self.feature} <  {np.round(self.threshold,2)}'
            right_rule = f'if {self.feature} >=  {np.round(self.threshold,2)}'
            if self.default_split == 'middle':
                middle_rule = f'if {self.feature} n/a'
            elif self.default_split=='left':
                left_rule += ' or n/a'
            else:
                right_rule += ' or n/a'

            print(hspace+f'{left_rule}:')
            self.left.print()
            if self.default_split == 'middle':
                print(hspace+f'{middle_rule}:')
                self.middle.print()
            print(hspace+f'{right_rule}:')
            self.right.print()

if __name__ == '__main__':
    seed = 12
    np.random.seed(seed)
    n = 1000 # number of data points
    p = 5 # Covariate dimension

    # Feature vector
    X = np.random.normal(0,1,(n,p))

    # Response dependence on covariates
    beta = np.arange(p)
    mu = X @ beta

    # Reponse
    y = np.random.normal(mu)

    # Missing value share
    missing_fraction = 0.5
    missing_index = np.random.binomial(1,missing_fraction,X.shape) == 1
    X[missing_index] = np.nan

    # Test train split
    test_index = np.random.choice(np.arange(n),int(n/10))
    X_train, X_test = X[~test_index], X[test_index]
    y_train, y_test = y[~test_index], y[test_index]

    # Tree hyperparameters
    max_depth = 4
    min_samples_split = 10

    # Create trees
    tree_maj = RegressionTree(max_depth = max_depth,
                                    min_samples_split = min_samples_split,
                                    missing_rule = 'majority')
    tree_mia = RegressionTree(max_depth = max_depth,
                                    min_samples_split = min_samples_split,
                                    missing_rule = 'mia')
    tree_tri = RegressionTree(max_depth = max_depth,
                                    min_samples_split = min_samples_split,
                                    missing_rule = 'trinary')
    tree_maj.fit(X_train,y_train)
    tree_mia.fit(X_train,y_train)
    tree_tri.fit(X_train,y_train)

    # Train data sse
    y_train_hat_maj = tree_maj.predict(X_train)
    y_train_hat_mia = tree_mia.predict(X_train)
    y_train_hat_tri = tree_tri.predict(X_train)
    sse_train_maj = sum((y_train_hat_maj - y_train)**2)
    sse_train_mia = sum((y_train_hat_mia - y_train)**2)
    sse_train_tri = sum((y_train_hat_tri - y_train)**2)
    print(pd.Series(data = [sse_train_maj,sse_train_mia,sse_train_tri],
              index = ['majority','mia','trinary'])/len(y_train))

    # Test data sse
    y_test_hat_maj = tree_maj.predict(X_test)
    y_test_hat_mia = tree_mia.predict(X_test)
    y_test_hat_tri = tree_tri.predict(X_test)
    sse_test_maj = sum((y_test_hat_maj - y_test)**2)
    sse_test_mia = sum((y_test_hat_mia - y_test)**2)
    sse_test_tri = sum((y_test_hat_tri - y_test)**2)
    print(pd.Series(data = [sse_test_maj,sse_test_mia,sse_test_tri],
              index = ['majority','mia','trinary'])/len(y_test))