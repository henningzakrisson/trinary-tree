# Trinary Tree

[![PyPI version](https://badge.fury.io/py/trinary_tree.svg)](https://pypi.org/project/trinary_tree/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/henningzakrisson/trinary_tree/blob/main/LICENSE)

The Trinary Tree is a algorithm based on the Classification and Regression Tree (CART).
It provides a novel way to handle missing data by assigning missing values to a third node in the originally binary 
split of the CART.
For details on the algorithm, se the arXiV preprint at https://arxiv.org/abs/2309.03561

## Installation

You can install the `trinary_tree` package via pip:

```bash
pip install trinary_tree
```
or via GitHub
````bash
pip install git+https://github.com/henningzakrisson/trinary_tree.git
````

## Usage example
Fitting a Trinary Tree and a Binary Tree using the majority
rule algorithm to a dataset with missing values.

````python
# Import packages
from src.trinary_tree import BinaryTree, TrinaryTree
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
rng = np.random.default_rng(seed=11)
X = rng.normal(size=(1000,2))
mu = 10*(X[:,0]>0) + X[:,1]*2
y = rng.normal(mu,1)

# Censor data
censor = rng.choice(np.prod(X.shape), int(0.2*np.prod(X.shape)), replace=False)
X_censored = X.flatten()
X_censored[censor] = np.nan
X_censored = X_censored.reshape(X.shape)

# Train trees
X_train, X_test, y_train, y_test = train_test_split(X_censored,y)
tree_binary = BinaryTree(max_depth = 1)
tree_trinary = TrinaryTree(max_depth = 1)
tree_binary.fit(X_train,y_train)
tree_trinary.fit(X_train,y_train)

# Calculate MSE
mse_binary = np.mean((y_test - tree_binary.predict(X_test))**2)
mse_trinary = np.mean((y_test - tree_trinary.predict(X_test))**2)
print(f"Binary tree MSE: {mse_binary:.3f}")
print(f"Trinary tree MSE: {mse_trinary:.3f}")
````

## Contact
If you have any questions, feel free to contact me
[here](mailto:henning.zakrisson@gmail.com).

