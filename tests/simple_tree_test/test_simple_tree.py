import unittest
import pandas as pd

class RegressionTreeTest(unittest.TestCase):
    def test_responses(self):
        from src.simple_tree.simple_tree import RegressionTree
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
        from src.simple_tree.simple_tree import RegressionTree
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
        from src.simple_tree.simple_tree import RegressionTree
        df = pd.read_csv('data/test_data_majority.csv',index_col = 0)

        max_depth = 2
        tree = tree = RegressionTree(max_depth = max_depth)
        tree.fit(df.loc[~df['y'].isna(),['X_0','X_1']],df.loc[~df['y'].isna(),'y'])

        df['y_hat'] = tree.predict(df[['X_0','X_1']])

        self.assertEqual((df.loc[~df['y'].isna(),'y_hat'].round(1)==df.loc[~df['y'].isna(),'y_hat_exp'].round(1)).sum().sum(),
                         sum(~df['y'].isna()))

    def test_default_mia(self):
        from src.simple_tree.simple_tree import RegressionTree
        df = pd.read_csv('data/test_data_mia.csv',index_col = 0)

        max_depth = 2
        tree = tree = RegressionTree(max_depth = max_depth,
                                     missing_rule='mia')
        tree.fit(df.loc[~df['y'].isna(),['X_0','X_1']],df.loc[~df['y'].isna(),'y'])

        df['y_hat'] = tree.predict(df[['X_0','X_1']])

        self.assertEqual((df['y_hat']==df['y']).sum(),len(df))

if __name__ == '__main__':
    unittest.main()
