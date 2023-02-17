import unittest
import pandas as pd
import numpy as np
from src.regression_tree.regression_tree import RegressionTree

from src.exceptions_and_warnings.custom_warnings import MissingFeatureWarning,ExtraFeatureWarning

class RegressionTreeTest(unittest.TestCase):
    def test_responses(self):
        df = pd.read_csv('data/test_data.csv',index_col = 0)

        df_hat = df[['y','X_0','X_1']].copy()
        for max_depth in [1,2]:
            tree = RegressionTree(max_depth = max_depth)
            tree.fit(df[['X_0','X_1']],df['y'])
            y_hat = tree.predict(df_hat[['X_0','X_1']])

            df_hat.loc[:,f'tree_{max_depth}'] = y_hat.copy()

        columns = [f'tree_{max_depth}' for max_depth in [1,2]]
        self.assertEqual((df[columns].round(3)==df_hat[columns].round(3)).sum().sum(),
                         len(columns) * len(df))

    def test_no_more_splits(self):
        df = pd.read_csv('data/test_data.csv',index_col = 0)

        df_hat = df[['y','X_0','X_1']].copy()
        max_depth = 10
        tree = RegressionTree(max_depth = max_depth)
        tree.fit(df[['X_0','X_1']],df['y'])
        y_hat = tree.predict(df_hat[['X_0','X_1']])

        df_hat.loc[:,f'tree_{max_depth}'] = y_hat.copy()

        self.assertEqual((df_hat[f'tree_{max_depth}'].round(3)==df[f'tree_{2}'].round(3)).sum().sum(),
                         len(df))

    def test_default_majority(self):
        df = pd.read_csv('data/test_data_majority.csv',index_col = 0)

        max_depth = 2
        tree = RegressionTree(max_depth = max_depth)
        tree.fit(df.loc[~df['y'].isna(),['X_0','X_1']],df.loc[~df['y'].isna(),'y'])

        df['y_hat'] = tree.predict(df[['X_0','X_1']])

        self.assertEqual((df.loc[~df['y'].isna(),'y_hat'].round(1)==df.loc[~df['y'].isna(),'y_hat_exp'].round(1)).sum().sum(),
                         sum(~df['y'].isna()))

    def test_default_mia(self):
        df = pd.read_csv('data/test_data_mia.csv',index_col = 0)

        max_depth = 2
        tree = RegressionTree(max_depth = max_depth,
                                     missing_rule='mia')
        tree.fit(df.loc[~df['y'].isna(),['X_0','X_1']],df.loc[~df['y'].isna(),'y'])

        df['y_hat'] = tree.predict(df[['X_0','X_1']])

        self.assertEqual((df['y_hat']==df['y']).sum(),len(df))

    def test_feature_importance(self):
        x0 = np.arange(1,100)
        x1 = np.ones(len(x0))
        X = np.stack([x0,x1]).T
        y = 10 * (x0>50)

        tree = RegressionTree(max_depth=2)
        tree.fit(X,y)
        feature_importance = tree.feature_importance()

        self.assertEqual(feature_importance[0],1)
        self.assertEqual(feature_importance[1],0)

    def test_missing_feature_warning(self):
        x0 = np.arange(1,100)
        x1 = np.ones(len(x0))
        x2 = np.tile(np.arange(0,10),10)[:-1]
        X = np.stack([x0,x1,x2]).T
        y = 10 * (x0>50) + 2 * (x2>5)

        tree = RegressionTree(max_depth=2)
        tree.fit(X,y)

        with self.assertWarns(MissingFeatureWarning):
            y_hat = tree.predict(X[:,:2])

    def test_redundant_feature_warning(self):
        x0 = np.arange(1,100)
        x1 = np.ones(len(x0))
        x2 = np.tile(np.arange(0,10),10)[:-1]
        X = np.stack([x0,x1]).T
        y = 10 * (x0>50) + 2

        tree = RegressionTree(max_depth=2)
        tree.fit(X,y)

        with self.assertWarns(ExtraFeatureWarning):
            X_extra = np.c_[X,x2]
            y_hat = tree.predict(X_extra)

    def test_trinary_tree(self):
        df_train = pd.read_csv('data/train_data_trinary.csv',index_col = 0)
        df_test = pd.read_csv('data/test_data_trinary.csv',index_col=0)

        tree = RegressionTree(max_depth = 2, missing_rule= 'trinary')
        tree.fit(df_train[['X_0','X_1']],df_train['y'])

        df_test['y_hat'] = tree.predict(df_test[['X_0','X_1']])

        self.assertEqual((df_test['y']==df_test['y_hat']).sum(),len(df_test))

if __name__ == '__main__':
    unittest.main()
