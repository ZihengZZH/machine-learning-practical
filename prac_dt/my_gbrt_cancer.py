from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import graphviz

import numpy as np
import matplotlib.pyplot as plt


print("\nloading the breast cancer dataset ...")
cancer = load_breast_cancer()

print("\nsplitting the training/test dataset ...")
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

def GBRT_classifier(randomState=0, maxDepth=0, learnRate=0):
    print("\ntraining a Gradient Boosting Tree classifier ...")
    if maxDepth != 0 and learnRate != 0:
        gbrt = GradientBoostingClassifier(random_state=randomState, max_depth=maxDepth, learning_rate=learnRate)
    elif maxDepth != 0:
        gbrt = GradientBoostingClassifier(random_state=randomState, max_depth=maxDepth)
    elif learnRate != 0:
        gbrt = GradientBoostingClassifier(random_state=randomState, learning_rate=learnRate)
    else:
        gbrt = GradientBoostingClassifier(random_state=randomState)
    gbrt.fit(X_train, y_train)

    print("\ntesting the Gradient Boosting Tree classifier ...")
    print("Accuracy on training set: %.3f" % gbrt.score(X_train, y_train))
    print("Accuracy on test set: %.3f" % gbrt.score(X_test, y_test))
    return gbrt

'''
As the training set accuracy is 100%, we are likely to be overfitting.
To reduce overfitting, we could either apply stronger pre-pruning by limiting the maximum depth or lower the learning rate.
TYPICALLY, learning rate = .01 or .001; depth of tree (d) = 1
'''

# GBRT_classifier()
# GBRT_classifier(maxDepth=1)
# GBRT_classifier(learnRate=.01)
# GBRT_classifier(maxDepth=1, learnRate=.01)

gbrt = GBRT_classifier(maxDepth=1)

n_features = cancer.data.shape[1]
plt.barh(range(n_features), gbrt.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.show()