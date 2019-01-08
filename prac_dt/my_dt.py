from sklearn.datasets import load_breast_cancer
from sklearn import tree
from sklearn.model_selection import train_test_split
import graphviz

import numpy as np
import matplotlib.pyplot as plt


def dt_classifier():
    print("\nbuilding a simple decision tree classifier ...")
    X = [[0,0], [1,1]]
    Y = [0, 1]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)
    predicted = clf.predict([[2.,2.]])
    print("prediction: ", predicted)
    predicted_prob = clf.predict_proba([[2.,2.]])
    print("prediction probability: ", predicted_prob)


def dt_regressor():
    print("\nbuilding a simple decision tree regressor ...")
    X = [[0,0], [2,2]]
    y = [0.5, 2.5]
    clf = tree.DecisionTreeRegressor()
    clf.fit(X, y)
    predicted = clf.predict([[1,1]])
    print("prediction: ", predicted)


def dt_cancer():
    print("\nloading the breast cancer dataset ...")
    cancer = load_breast_cancer()

    print("\nsplitting the training/test dataset ...")
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    print("\ntraining a dt classifier w/o pruning ...")
    tree_cancer = tree.DecisionTreeClassifier(random_state=0)
    tree_cancer.fit(X_train, y_train)
    print("Accuracy on training set: {0:.3f}".format(tree_cancer.score(X_train, y_train)))
    print("Accuracy on test set: {0:.3f}".format(tree_cancer.score(X_test, y_test)))

    print("\ntraining a dt classifier w/ pruning ...")
    tree_cancer_prune = tree.DecisionTreeClassifier(max_depth=4, random_state=0)
    tree_cancer_prune.fit(X_train, y_train)
    print("Accuracy on training set: {0:.3f}".format(tree_cancer_prune.score(X_train, y_train)))
    print("Accuracy on test set: {0:.3f}".format(tree_cancer_prune.score(X_test, y_test)))

    print("\nObviously, pruning somehow avoids the overfitting.")

    tree.export_graphviz(tree_cancer_prune, out_file="tree.dot", class_names=["malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    graphviz.Source(dot_graph)

    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), tree_cancer_prune.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.show()


if __name__ == "__main__":
    # dt_classifier()
    # dt_regressor()
    dt_cancer()