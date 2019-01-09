from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import graphviz

import numpy as np
import matplotlib.pyplot as plt


print("\nloading the breast cancer dataset ...")
cancer = load_breast_cancer()

print("\nsplitting the training/test dataset ...")
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

print("\ntraining a Random Forest classifier ...")
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("\ntesting the Random Forest classifier ...")
print("Accuracy on training set: %.3f" % forest.score(X_train, y_train))
print("Accuracy on test set: %.3f" % forest.score(X_test, y_test))


n_features = cancer.data.shape[1]
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()