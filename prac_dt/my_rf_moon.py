import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mglearn


# load the data
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
print("\ndescription of the data ...")
print("X\n", X[:10])
print("y\n", y[:10])

# training / test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# train the model
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

# draw the tree plots
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree %d" % i)
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:,0], X_train[:,1], y_train)
plt.savefig("./trees.png", dpi=300)

