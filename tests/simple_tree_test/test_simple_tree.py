import unittest
import pandas as pd

class TreeRegressionTest(unittest.TestCase):
    def test_responses(self):
        from src.simple_tree.simple_tree import regressionTree
        df = pd.read_csv('test_data.csv',index_col = 0)

        df_hat = df[['y','X_0','X_1']].copy()
        for max_depth in [1,2,3,4]:
            tree = regressionTree(max_depth = max_depth)
            tree.fit(df[['X_0','X_1']],df['y'])
            y_hat = tree.predict(df_hat[['X_0','X_1']])

            df_hat.loc[:,f'tree_{max_depth}'] = y_hat.copy()

        columns = [f'tree_{max_depth}' for max_depth in [1,2,3,4]]
        self.assertEqual((df[columns].round(4)==df_hat[columns].round(4)).sum().sum(),
                         len(columns) * len(df))


if __name__ == '__main__':
    unittest.main()
