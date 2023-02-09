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
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth

    def fit(self, X, y):
        """Recursive method to fit the decision tree"""
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.Series:
            y = y.values
        self.yhat = y.mean()
        self.mse = ((y-self.yhat)**2).mean()
        self.n = len(y)

        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_split):
            self.left = None
            self.right = None
        else:
            self.feature, self.threshold = self.find_split(X, y)
            if self.feature is None:
                self.left = None
                self.right = None
            else:
                index_left = X[:,self.feature]<self.threshold

                self.left = RegressionTree(
                            min_samples_split=self.min_samples_split,
                            max_depth=self.max_depth,
                            depth=self.depth + 1,
                            )
                self.right = copy.copy(self.left)

                self.left.fit(X[index_left], y[index_left])
                self.right.fit(X[~index_left], y[~index_left])

    def find_split(self, X, y) -> tuple:
        """Calculate the best split for a decision tree"""
        mse_best = self.mse
        best_feature = None
        best_threshold = None

        for feature in range(0,X.shape[1]):
            values = np.sort(np.unique(X[:,feature]))
            numbers_between_values = np.convolve(values, np.ones(2), 'valid') / 2
            threshold_candidates = numbers_between_values[1:-1]

            for threshold in threshold_candidates:
                index_left = X[:,feature]<threshold

                if (sum(index_left)>=self.min_samples_split) and (sum(~index_left)>=self.min_samples_split):
                    sse_left = ((y[index_left] - y[index_left].mean())**2).sum()
                    sse_right = ((y[~index_left] - y[~index_left].mean())**2).sum()
                    mse = (sse_left + sse_right)/len(y)
                    if mse < mse_best:
                        mse_best = mse
                        best_feature = feature
                        best_threshold = threshold

        return (best_feature, best_threshold)

    def predict(self,X):
        """Recursive method to predict from new of features"""
        if type(X) == pd.DataFrame:
            X = X.values

        if self.left is None:
            y_hat = np.ones(len(X))*self.yhat
            return y_hat
        else:
            # Index split
            index_left = X[:,self.feature]<self.threshold
            y_hat = np.ones(len(X))
            y_hat[index_left] = self.left.predict(X[index_left])
            y_hat[~index_left] =  self.right.predict(X[~index_left])

            return y_hat

    def print(self):
        hspace = '---'*self.depth
        print(hspace+f'Number of observations: {self.n}')
        print(hspace+f'Response estimate: {np.round(self.yhat,2)}')
        if self.left is not None:
            print(hspace+f'if {self.feature} <  {np.round(self.threshold,2)}:')
            self.left.print()

            print(hspace+f'if {self.feature} >= {np.round(self.threshold,2)}:')
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
