# Data handling and math
import pandas as pd
import numpy as np

class regressionTree():
    """
    Class to grow a regression decision tree
    """
    def __init__(
        self,
        min_samples_split=20,
        max_depth=2,
        depth=0,
        node_type='root',
        rule=""
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.node_type = node_type
        self.rule = rule

    def fit(self,X,y):
        """
        Recursive method to fit the decision tree
        """
        # Store data in node
        if type(X) == pd.DataFrame:
            self.X = X
        else:
            self.X = pd.DataFrame(X, columns = [f'feature_{i}' for i in range(0,X.shape[1])])
        self.y = y
        self.ymean = self.y.mean()
        self.n = len(y)

        # Check terminal node conditions
        if (self.depth >= self.max_depth) or (self.n <= self.min_samples_split):
            self.left = None
            self.right = None
            self.node_type = 'terminal'
        else:
            # Find best split
            df = X.copy()
            df['y'] = y
            self.splitting_feature, self.threshold = self.find_best_split(df.drop('y',axis = 1),
                                                                     df['y'])

            # Split data
            df_left = df[df[self.splitting_feature]<=self.threshold].copy()
            df_right = df[df[self.splitting_feature]>self.threshold].copy()

            # Create left node
            self.left = regressionTree(
                        min_samples_split=self.min_samples_split,
                        max_depth=self.max_depth,
                        depth=self.depth + 1,
                        node_type='left_node',
                        rule=f"{self.splitting_feature} < {round(self.threshold, 3)}"
                        )
            self.left.fit(df_left.drop('y',axis = 1),
                          df_left['y'])

            # Create right node
            self.right = regressionTree(
                        min_samples_split=self.min_samples_split,
                        max_depth=self.max_depth,
                        depth=self.depth + 1,
                        node_type='right_node',
                        rule=f"{self.splitting_feature} >= {round(self.threshold, 3)}"
                        )
            self.right.fit(df_right.drop('y',axis = 1),
                           df_right['y'])

    def find_best_split(self,X,y) -> tuple:
        """
        Given the X features and y targets calculates the best split
        for a decision tree
        """
        # Create a dataset for spliting
        df = X.copy()
        df['y'] = y

        mse_best = np.inf
        best_splitting_feature = X.columns[0]
        best_threshold = X[best_splitting_feature].max()+1

        for feature in X.columns:
            # Get candidate values
            values = np.sort(X[feature].copy().drop_duplicates().values)
            numbers_between_values = np.convolve(values, np.ones(2), 'valid') / 2
            threshold_candidates = numbers_between_values[1:-1]

            for threshold in threshold_candidates:
                # Getting the left and right ys
                y_left  = df.loc[df[feature]<threshold,'y']
                y_right = df.loc[df[feature]>=threshold,'y']

                # Check if this leads to too few values in the daughter nodes
                if (len(y_left)>= self.min_samples_split) and (len(y_right)>= self.min_samples_split):
                    # Calculate mean squared error
                    mse_left = np.mean((y_left - y_left.mean())**2)
                    mse_right = np.mean((y_right - y_right.mean())**2)
                    mse_split = mse_left + mse_right

                    # Checking if this is the best split so far
                    if mse_split < mse_best:
                        best_splitting_feature = feature
                        best_threshold = threshold

                        # Setting the best gain to the current one
                        mse_best = mse_split

        return (best_splitting_feature, best_threshold)

    def predict(self,XTest):
        """
        Recursive method to predict from new of features
        """
        # Making a df from the data
        df = XTest.copy()
        nTest = len(df)

        # If terminal node - output value
        if self.left is None:
            df['Y'] = self.ymean
            return df['Y']
        else:
            # For left-goers
            yhat_left = self.left.predict(df.loc[df[self.splitting_feature]<self.threshold])
            df.loc[df[self.splitting_feature]<self.threshold,'Y'] = yhat_left

            # For right-goers
            yhat_right = self.right.predict(df.loc[df[self.splitting_feature]>=self.threshold])
            df.loc[df[self.splitting_feature]>=self.threshold,'Y'] = yhat_right

            return df['Y']

    def print_info(self, width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const

        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")

    def print_tree(self):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info()

        if self.left is not None:
            self.left.print_tree()

        if self.right is not None:
            self.right.print_tree()
