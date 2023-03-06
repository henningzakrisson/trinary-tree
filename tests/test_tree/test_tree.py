import unittest
import pandas as pd
import numpy as np
import warnings
from src.trinary_tree import TrinaryTree
from src.binary_tree import BinaryTree
from src.weighted_tree import WeightedTree

from src.common.custom_warnings import (
    MissingFeatureWarning,
    ExtraFeatureWarning,
)

from src.common.functions import (
get_feature_importance
)

class TreeTest(unittest.TestCase):
    """Module to test the functionality of the regression trees"""

    def test_responses(self):
        """Basic test of the default settings of the tree"""
        df = pd.read_csv("data/test_data.csv", index_col=0)

        max_depths = [1,2]
        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}

        for max_depth in max_depths:
            for tree_name in tree_types:
                tree = tree_types[tree_name](max_depth=max_depth)
                tree.fit(df[["X_0", "X_1"]], df["y"])
                y_hat = tree.predict(df[["X_0", "X_1"]])

                self.assertEqual(
                    (df[f"tree_{max_depth}"].round(3) == y_hat.round(3)).sum(),
                        len(df),
                    msg=f"Response prediction not equal to expected response for {tree_name} of depth {max_depth}",
                )

    def test_no_more_splits(self):
        """Test so that the tree doesn't make any unnecessary splits if no more
        information is go be gained"""
        df = pd.read_csv("data/test_data.csv", index_col=0)

        max_depth = 10
        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}

        for tree_name in tree_types:
            tree = tree_types[tree_name](max_depth=max_depth)
            tree.fit(df[["X_0", "X_1"]], df["y"])
            y_hat = tree.predict(df[["X_0", "X_1"]])

            self.assertEqual(
                (df["tree_2"].round(3) == y_hat.round(3)).sum(),
                len(df),
                msg=f"Response prediction not equal to expected response for {tree_name}, possibly making to many splits.",
            )

    def test_default_majority(self):
        """Test majority rule handling of missing values"""
        df = pd.read_csv("data/test_data_majority.csv", index_col = 0)
        df = df.loc[~df['y'].isna()]

        max_depth = 2
        tree = BinaryTree(max_depth = max_depth, missing_rule = 'majority')
        tree.fit(df[["X_0", "X_1"]], df["y"])

        y_hat = tree.predict(df[["X_0", "X_1"]])

        self.assertEqual(
            (
                y_hat.round(1)
                == df["y_hat_exp"].round(1)
            )
            .sum(),
            len(df),
            msg="Reponse prediction for majority rule strategy not as expected",
        )

    def test_default_mia(self):
        """Test MIA handling of missing values"""
        df = pd.read_csv("data/test_data_mia.csv", index_col=0)
        df = df.loc[~df['y'].isna()]

        max_depth = 2
        tree = BinaryTree(max_depth=max_depth, missing_rule="mia")
        tree.fit(df[["X_0", "X_1"]], df["y"])

        y_hat = tree.predict(df[["X_0", "X_1"]])

        self.assertEqual(
            (
                    y_hat.round(1)
                    == df["y"].round(1)
            )
            .sum(),
            len(df),
            msg="Reponse prediction for majority rule strategy not as expected",
        )

    def test_trinary_tree(self):
        """Test trinary tree handling of missing values"""
        df_train = pd.read_csv("data/train_data_trinary.csv", index_col=0)
        df_test = pd.read_csv("data/test_data_trinary.csv", index_col=0)

        tree = TrinaryTree(max_depth=2)
        tree.fit(df_train[["X_0", "X_1"]], df_train["y"])

        df_test["y_hat"] = tree.predict(df_test[["X_0", "X_1"]])

        self.assertEqual(
            (df_test["y"] == df_test["y_hat"]).sum(),
            len(df_test),
            msg="Response prediction for trinary tree not as expected",
        )

    def test_tree_cat(self):
        """Test for categorical input"""
        df = pd.read_csv("data/test_data_cat.csv", index_col=0)
        X = df.drop("y", axis=1)
        y = df["y"]

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        max_depth = 4
        min_samples_leaf = 1

        for tree_name in tree_types:
            tree = tree_types[tree_name](max_depth=max_depth, min_samples_leaf =min_samples_leaf)
            tree.fit(X,y)
            y_hat = tree.predict(X)

            self.assertEqual(
                (y == y_hat).sum(),
                len(y),
                msg=f"Categorical data prediction not correct for {tree_name}",
            )

    def test_feature_importance(self):
        """Test feature importance values of very simple example"""
        x0 = np.arange(1, 100)
        x1 = np.ones(len(x0))
        X = np.stack([x0, x1]).T
        y = 10 * (x0 > 50)

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        max_depth = 2

        for tree_name in tree_types:
            tree = tree_types[tree_name](max_depth=max_depth)
            tree.fit(X, y)

            feature_importance = get_feature_importance(tree)

            self.assertEqual(
                feature_importance[0],
                1,
                msg=f"Feature importance of relevant feature not correct for {tree_name}",
            )
            self.assertEqual(
                feature_importance[1],
                0,
                msg=f"Feature importance of irrelevant feature not correct for {tree_name}",
            )

    def test_missing_feature_warning(self):
        """Check so that a warning is thrown when trying to predict with a insufficient covariate vector"""
        x0 = np.arange(1, 100)
        x1 = np.ones(len(x0))
        x2 = np.tile(np.arange(0, 10), 10)[:-1]
        X = np.stack([x0, x1, x2]).T
        y = 10 * (x0 > 50) + 2 * (x2 > 5)

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        for tree_name in tree_types:
            tree = tree_types[tree_name]()
            tree.fit(X, y)

            with self.assertWarns(
                MissingFeatureWarning, msg=f"Missing feature warning missing for {tree_name}"
            ):
                y_hat = tree.predict(X[:, :2])

    def test_redundant_feature_warning(self):
        """Check so that a warning is thrown when trying to predict with a covariate vector with irrelevant features"""
        x0 = np.arange(1, 100)
        x1 = np.ones(len(x0))
        x2 = np.tile(np.arange(0, 10), 10)[:-1]
        X = np.stack([x0, x1]).T
        y = 10 * (x0 > 50) + 2
        X_extra = np.c_[X, x2]

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        for tree_name in tree_types:
            tree = tree_types[tree_name]()
            tree.fit(X, y)

            with self.assertWarns(
                ExtraFeatureWarning, msg=f"Redundant feature warning missing for {tree_name}"
            ):
                y_hat = tree.predict(X_extra)

    def test_classification(self):
        df = pd.read_csv("data/test_data_class.csv", index_col=0)
        X = df.drop("y", axis=1)
        y = df["y"]

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        max_depth = 3
        min_samples_leaf = 20
        for tree_name in tree_types:
            tree = tree_types[tree_name](max_depth = max_depth, min_samples_leaf = min_samples_leaf)
            tree.fit(X, y)
            df["y_hat"] = tree.predict(X)

            self.assertEqual(
                (df["y"] == df["y_hat"]).sum(),
                len(df),
                msg=f"Classification not identical to expected for {tree_name}",
            )

    def test_classification_probabilities(self):
        df = pd.read_csv("data/test_data_proba.csv", index_col=0)

        X = df[["feature"]]
        y = df["y"]

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        max_depth = 2
        for tree_name in tree_types:
            tree = tree_types[tree_name](max_depth = 2)
            tree.fit(X, y)


            msg = f"Wrong probability for {tree_name}"

            self.assertAlmostEqual(tree.y_prob["banana"], 0.5, msg=msg)
            self.assertAlmostEqual(tree.y_prob["apple"], 0.5, msg=msg)
            self.assertAlmostEqual(
                tree.right.y_prob["apple"], 5 / 6, msg=msg
            )
            self.assertAlmostEqual(
                tree.right.left.y_prob["apple"], 1, msg=msg
            )
            self.assertAlmostEqual(
                tree.right.right.y_prob["banana"], 1, msg=msg
            )

    def test_probability_predictions(self):
        df = pd.read_csv("data/test_data_class.csv", index_col=0)
        X = df.drop("y", axis=1)
        y = df["y"]

        tree_types = {'Binary': BinaryTree, 'Trinary': TrinaryTree, 'Weighted': WeightedTree}
        max_depth = 2
        min_samples_leaf = 20

        for tree_name in tree_types:
            tree = tree_types[tree_name](max_depth = max_depth, min_samples_leaf = min_samples_leaf)
            tree.fit(X, y)

            y_hat = tree.predict(X)
            y_probs = tree.predict(X, prob=True)

            self.assertEqual(
                (y_probs.idxmax(axis=1) == y_hat).sum(),
                len(df),
                msg=f"Most probable category not predicted for {tree_name}",
            )

    def test_weighted_strat(self):
        df_train = pd.read_csv('data/train_data_weighted.csv',index_col=0)
        X_train = df_train.drop('y',axis=1)
        y_train = df_train['y']

        df_test =  pd.read_csv('data/test_data_weighted.csv',index_col=0)
        X_test = df_test.drop('y', axis=1)
        y_test = df_test['y']

        tree = WeightedTree(max_depth=2, min_samples_leaf = 1)
        tree.fit(X_train,y_train)

        y_hat = tree.predict(X_test)

        self.assertAlmostEqual((y_hat==y_test).sum(),len(y_test),msg="Weighted strategy not working properly")

    def test_weighted_strat_cat(self):
        df_train = pd.read_csv('data/train_data_weighted_cat.csv',index_col=0)
        X_train = df_train.drop('y',axis=1)
        y_train = df_train['y']

        df_test =  pd.read_csv('data/test_data_weighted_cat.csv',index_col=0)
        X_test = df_test[['number','fruit']]
        y_prob = df_test[['bad','good','great']]
        y_test = df_test['y']

        tree = WeightedTree(max_depth=2, min_samples_leaf = 1)
        tree.fit(X_train,y_train)

        y_prob_hat = tree.predict(X_test,prob = True)
        y_hat = tree.predict(X_test)

        self.assertAlmostEqual((y_hat==y_test).sum(),len(y_test),msg="Weighted strategy classification not working properly")
        self.assertAlmostEqual((y_prob_hat.round(2)==y_prob.round(2)).sum().sum(),np.prod(y_prob.shape),msg="Weighted strategy classification not working properly")

if __name__ == "__main__":
    unittest.main()
