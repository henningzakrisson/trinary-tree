import numpy as np
import pandas as pd
from src.regression_tree import RegressionTree

# Create data
n = 1000

X = pd.DataFrame(index = range(n))
X['cont_normal'] = np.random.normal(3,2,n)
X['cont_linear'] = np.arange(n)
X['cont_stairs'] = np.concatenate([np.ones(int(i)) * i for i in np.linspace(0,201,10)])[:n]
X['cat_version'] = np.random.choice(['basic','pro','extra','none'],n,p = [0.5,0.22,0.18,0.1])
X['cat_gender'] = np.random.choice(['male','female'],n,p = [0.59,1-0.59])

# True tree structure
left_00 = X['cont_normal']<X['cont_normal'].quantile(0.7)
left_10 = X['cat_version'].isin(['basic','extra'])
left_11 = X['cat_gender']=='male'
left_20 = X['cont_stairs'] < X['cont_stairs'].quantile(0.4)
left_21 = X['cont_stairs'] < X['cont_stairs'].quantile(0.4)
left_22 = X['cat_version'].isin(['basic','none'])
left_23 = X['cont_linear'] < X['cont_linear'].mean()

index_30 = left_00 & left_10 & left_20
index_31 = left_00 & left_10 & (~left_20)
index_32 = left_00 & (~left_10) & left_21
index_33 = left_00 & (~left_10) & (~left_21)
index_34 = (~left_00) & left_11 & left_22
index_35 = (~left_00) & left_11 & (~left_22)
index_36 = (~left_00) & (~left_11) & left_23
index_37 = (~left_00) & (~left_11) & (~left_23)

terminal_node_indices = [index_30, index_31, index_32, index_33, index_34, index_35, index_36, index_37]
mus = np.arange(8)*10

y = pd.Series(index = X.index, dtype ='float')
for index,mu in zip(terminal_node_indices,mus):
    y.loc[index] = np.random.normal(mu,1)

# Hyperparameters
max_depth = 3
min_samples_split = 5
strategies =['majority','mia','trinary']

# Example 1: No missing data

# Train-test-split
index_train = [i in np.random.choice(X.index, int(0.8*n)) for i in X.index]
index_test = [(1-i) == 1 for i in index_train]
X_train, y_train = X.loc[index_train], y.loc[index_train]
X_test, y_test = X.loc[index_test], y.loc[index_test]

# Fit trees
tree = RegressionTree()
tree.fit(X_train,y_train)
