# Data handling and math
import pandas as pd
import numpy as np
import copy


class RegressionTree:
    """
    Class to grow a regression decision tree
    """
    def __init__(
        self,
        min_samples_split=20,
        max_depth=2,
        depth=0,
        missing_rule = 'majority'
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.missing_rule = missing_rule

        self.n = 0
        self.yhat = None
        self.sse = None
        self.feature = None
        self.threshold = None
        self.default_split = None
        self.left = None
        self.right = None

    def fit(self, X, y):
        """Recursive method to fit the decision tree"""
        X = X.values if isinstance(X,pd.DataFrame) else X
        y = y.values if isinstance(y,pd.Series) else y

        X, y = X[~np.isnan(y)], y[~np.isnan(y)]

        self.yhat = y.mean()
        self.sse = ((y-self.yhat)**2).sum()
        self.n = len(y)

        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_split):
            return

        self.feature, self.threshold, self.default_split = self._find_split(X, y)
        if self.feature is None:
            return
        if self.default_split == 'left':
            index_left = (X[:,self.feature]<self.threshold)|np.isnan(X[:,self.feature])
        else:
            index_left = (X[:,self.feature]<self.threshold)

        self.left = self._initiate_daughter_node()
        self.right = self._initiate_daughter_node()

        self.left.fit(X[index_left], y[index_left])
        self.right.fit(X[~index_left], y[~index_left])

    def _find_split(self, X, y) -> tuple:
        """Calculate the best split for a decision tree"""
        sse_best = self.sse
        best_feature = None
        best_threshold = None
        best_default_split = 'right'

        for feature in range(0,X.shape[1]):
            values = np.sort(np.unique(X[:,feature]))
            numbers_between_values = np.convolve(values, np.ones(2), 'valid') / 2
            threshold_candidates = numbers_between_values[1:-1]

            for threshold in threshold_candidates:
                if self.missing_rule == 'majority' or (self.missing_rule == 'mia' and sum(np.isnan(X[:,feature]))==0):
                    default_split = 'left' if sum(X[:,feature]<threshold)>sum(X[:,feature]>=threshold) else 'right'
                    sse = self._calculate_split_sse(X,y,feature,threshold,default_split)
                    if sse < sse_best:
                        sse_best = sse
                        best_feature = feature
                        best_threshold = threshold
                        best_default_split = default_split

                elif self.missing_rule == 'mia':
                    for default_split in ['left','right']:
                        sse = self._calculate_split_sse(X,y,feature,threshold,default_split)
                        if sse < sse_best:
                            sse_best = sse
                            best_feature = feature
                            best_threshold = threshold
                            best_default_split = default_split

        return (best_feature, best_threshold, best_default_split)

    def _calculate_split_sse(self,X,y,feature,threshold,default_split):
        if default_split == 'left':
            index_left = (X[:,feature]<threshold)|np.isnan(X[:,feature])
        else:
            index_left = (X[:,feature]<threshold)

        # To avoid hyperparameter-illegal splits
        if (sum(index_left)<self.min_samples_split) or (sum(~index_left)<self.min_samples_split):
            return self.sse

        sse_left = ((y[index_left] - y[index_left].mean())**2).sum()
        sse_right = ((y[~index_left] - y[~index_left].mean())**2).sum()
        return sse_left + sse_right

    def _initiate_daughter_node(self):
        return RegressionTree(
                    min_samples_split=self.min_samples_split,
                    max_depth=self.max_depth,
                    depth=self.depth + 1,
                    missing_rule= self.missing_rule
                    )

    def predict(self,X):
        """Recursive method to predict from new of features"""
        X = X.values if isinstance(X,pd.DataFrame) else X
        y_hat = np.ones(len(X))*np.nan

        if self.left is None:
            y_hat[:] = self.yhat
            return y_hat

        if self.default_split == 'left':
            index_left = (X[:,self.feature]<self.threshold)|np.isnan(X[:,self.feature])
        else:
            index_left = (X[:,self.feature]<self.threshold)

        y_hat[index_left] = self.left.predict(X[index_left])
        y_hat[~index_left] = self.right.predict(X[~index_left])

        return y_hat

    def print(self):
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

    tree = RegressionTree(max_depth=5, min_samples_split = 1)
    tree.fit(X = X, y = y)

    tree.print()
