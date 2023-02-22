import unittest
import pandas as pd
import numpy as np
from src.tree import Tree

from src.exceptions_and_warnings.custom_warnings import (
    MissingFeatureWarning,
    ExtraFeatureWarning,
)


class TreeTest(unittest.TestCase):
    """Module to test the functionality of the regression trees"""

    def test_responses(self):
        """Basic test of the default settings of the tree"""
        df = pd.read_csv("data/test_data.csv", index_col=0)

        df_hat = df[["y", "X_0", "X_1"]].copy()
        for max_depth in [1, 2]:
            tree = Tree(max_depth=max_depth)
            tree.fit(df[["X_0", "X_1"]], df["y"])
            y_hat = tree.predict(df_hat[["X_0", "X_1"]])

            df_hat.loc[:, f"tree_{max_depth}"] = y_hat.copy()

        columns = [f"tree_{max_depth}" for max_depth in [1, 2]]
        self.assertEqual(
            (df[columns].round(3) == df_hat[columns].round(3)).sum().sum(),
            len(columns) * len(df),
            msg="Response prediction not equal to expected response",
        )

    def test_no_more_splits(self):
        """Test so that the tree doesn't make any unnecessary splits if no more
        information is go be gained"""
        df = pd.read_csv("data/test_data.csv", index_col=0)

        df_hat = df[["y", "X_0", "X_1"]].copy()
        max_depth = 10
        tree = Tree(max_depth=max_depth)
        tree.fit(df[["X_0", "X_1"]], df["y"])
        y_hat = tree.predict(df_hat[["X_0", "X_1"]])

        df_hat.loc[:, f"tree_{max_depth}"] = y_hat.copy()

        self.assertEqual(
            (df_hat[f"tree_{max_depth}"].round(3) == df[f"tree_{2}"].round(3))
            .sum()
            .sum(),
            len(df),
            msg="Response prediction not equal to expected response. Possibly making to many splits.",
        )

    def test_default_majority(self):
        """Test majority rule handling of missing values"""
        df = pd.read_csv("data/test_data_majority.csv", index_col=0)

        max_depth = 2
        tree = Tree(max_depth=max_depth)
        tree.fit(df.loc[~df["y"].isna(), ["X_0", "X_1"]], df.loc[~df["y"].isna(), "y"])

        df["y_hat"] = tree.predict(df[["X_0", "X_1"]])

        self.assertEqual(
            (
                df.loc[~df["y"].isna(), "y_hat"].round(1)
                == df.loc[~df["y"].isna(), "y_hat_exp"].round(1)
            )
            .sum()
            .sum(),
            sum(~df["y"].isna()),
            msg="Reponse prediction for majority rule strategy not as expected",
        )

    def test_default_mia(self):
        """Test MIA handling of missing values"""
        df = pd.read_csv("data/test_data_mia.csv", index_col=0)

        max_depth = 2
        tree = Tree(max_depth=max_depth, missing_rule="mia")
        tree.fit(df.loc[~df["y"].isna(), ["X_0", "X_1"]], df.loc[~df["y"].isna(), "y"])

        df["y_hat"] = tree.predict(df[["X_0", "X_1"]])

        self.assertEqual(
            (df["y_hat"] == df["y"]).sum(),
            len(df),
            msg="Reponse prediction for MIA strategy not as expected",
        )

    def test_trinary_tree(self):
        """Test trinary tree handling of missing values"""
        df_train = pd.read_csv("data/train_data_trinary.csv", index_col=0)
        df_test = pd.read_csv("data/test_data_trinary.csv", index_col=0)

        tree = Tree(max_depth=2, missing_rule="trinary")
        tree.fit(df_train[["X_0", "X_1"]], df_train["y"])

        df_test["y_hat"] = tree.predict(df_test[["X_0", "X_1"]])

        self.assertEqual(
            (df_test["y"] == df_test["y_hat"]).sum(),
            len(df_test),
            msg="Response prediction for trinary tree not as expected",
        )

    def test_tree_cat(self):
        """ Test for categorical input"""
        df = pd.read_csv("data/test_data_cat.csv", index_col=0)
        X = df.drop('y',axis=1)
        y = df['y']
        tree = Tree(max_depth = 4, min_samples_leaf = 1)
        tree.fit(X, y)
        y_hat = tree.predict(X)

        self.assertEqual((y==y_hat).sum(),len(y),msg = "Categorical data prediction not correct for all datapoints")

    def test_feature_importance(self):
        """Test feature importance values of very simple example"""
        x0 = np.arange(1, 100)
        x1 = np.ones(len(x0))
        X = np.stack([x0, x1]).T
        y = 10 * (x0 > 50)

        tree = Tree(max_depth=2)
        tree.fit(X, y)
        feature_importance = tree.feature_importance()

        self.assertEqual(
            feature_importance[0],
            1,
            msg="Feature importance of relevant feature not correct",
        )
        self.assertEqual(
            feature_importance[1],
            0,
            msg="Feature importance of irrelevant feature not correct",
        )

    def test_missing_feature_warning(self):
        """Check so that a warning is thrown when trying to predict with a insufficient covariate vector"""
        x0 = np.arange(1, 100)
        x1 = np.ones(len(x0))
        x2 = np.tile(np.arange(0, 10), 10)[:-1]
        X = np.stack([x0, x1, x2]).T
        y = 10 * (x0 > 50) + 2 * (x2 > 5)

        tree = Tree(max_depth=2)
        tree.fit(X, y)

        with self.assertWarns(
            MissingFeatureWarning, msg="Missing feature warning missing"
        ):
            y_hat = tree.predict(X[:, :2])

    def test_redundant_feature_warning(self):
        """Check so that a warning is thrown when trying to predict with a covariate vector with irrelevant features"""
        x0 = np.arange(1, 100)
        x1 = np.ones(len(x0))
        x2 = np.tile(np.arange(0, 10), 10)[:-1]
        X = np.stack([x0, x1]).T
        y = 10 * (x0 > 50) + 2

        tree = Tree(max_depth=2)
        tree.fit(X, y)

        with self.assertWarns(
            ExtraFeatureWarning, msg="Redundant feature warning missing"
        ):
            X_extra = np.c_[X, x2]
            y_hat = tree.predict(X_extra)


    def test_classification(self):
        df = pd.read_csv('data/test_data_class.csv',index_col=0)
        X = df.drop('y',axis=1)
        y = df['y']

        tree = Tree(max_depth = 3, min_samples_leaf=20)
        tree.fit(X,y)

        df['y_hat'] = tree.predict(X)

        self.assertEqual((df['y']==df['y_hat']).sum(),len(df),msg = "Classification not identical to expected")

    def test_classification_probabilities(self):
        df = pd.read_csv('data/test_data_proba.csv',index_col=0)

        X = df[['feature']]
        y = df['y']

        tree = Tree(max_depth=2)
        tree.fit(X, y)

        self.assertAlmostEqual(tree.y_prob['banana'],0.5, msg='Wrong probability')
        self.assertAlmostEqual(tree.y_prob['apple'], 0.5, msg='Wrong probability')
        self.assertAlmostEqual(tree.right.y_prob['apple'], 5/6, msg='Wrong probability')
        self.assertAlmostEqual(tree.right.left.y_prob['apple'],1, msg='Wrong probability')
        self.assertAlmostEqual(tree.right.right.y_prob['banana'],1, msg='Wrong probability')

    def test_probability_predictions(self):
        df = pd.read_csv('data/test_data_class.csv',index_col=0)
        X = df.drop('y',axis=1)
        y = df['y']

        tree = Tree(max_depth = 3, min_samples_leaf=20)
        tree.fit(X,y)

        df['y_hat'] = tree.predict(X)
        df_probs = tree.predict(X, prob = True)

        self.assertEqual((df_probs.idxmax(axis=1)==df['y_hat']).sum(),len(df),
                         msg = "Most probable category not predicted")


if __name__ == "__main__":
    unittest.main()
