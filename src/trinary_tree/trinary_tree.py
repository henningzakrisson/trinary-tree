import itertools
import pandas as pd
import numpy as np
import copy

from src.exceptions import MissingValuesInRespnonse,CantPrintUnfittedTree

class TrinaryRegressionTree:
    """
    Class to grow a trinary regression decision tree with missing-value handling
    """
    def __init__(
            self,
            min_samples_split = 20,
            max_depth = 2,
            depth = 0,
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

def fit(self, X, y):
        """Recursive method to fit the decision tree"""
        if np.any(np.isnan(y)):
            raise MissingValuesInRespnonse("n/a not allowed in response (y)")

        X = X.values if isinstance(X,pd.DataFrame) else X
        y = y.values if isinstance(y,pd.Series) else y
