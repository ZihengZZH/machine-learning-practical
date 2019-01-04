from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from svmPlot import svmPlot
import numpy as np

import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

iris = datasets.load_iris()

# load the iris dataset
# NOTE that we only use first two features for 2-d plot
X = iris.data[:,:2]
y = iris.target
indices = np.random.permutation(len(X))
test_size = 15
X_train = X[indices[:-test_size]]
y_train = y[indices[:-test_size]]
X_test = X[indices[-test_size:]]
y_test = y[indices[-test_size:]]


def linear_svm(para_c):
    # para para_c: error penalty term (controls the trade-off between weights and margins in the cost function)
    # type para_c: int
    SVM_linear = svm.SVC(kernel='linear', C=para_c) # linear kernel
    
    #train the SVM
    SVM_linear.fit(X_train, y_train)
    # test the SVM
    y_pred = SVM_linear.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))
    print("Overall accuracy: ", round(metrics.accuracy_score(y_test, y_pred), 2))

    # visualise the decision boundaries
    svmPlot(X, y, SVM_linear)


def apply_one_svm(name):
    # para name: the type of the SVM classifier
    # type name: str
    if name == 'linear':
        SVM_defined = svm.SVC(kernel='linear', C=1) 
    elif name == 'poly':
        SVM_defined = svm.SVC(kernel='poly', C=1, degree=3, coef0=1, gamma='scale') 
    elif name == 'rbf':
        SVM_defined = svm.SVC(kernel='rbf', C=1, gamma=1)
    else:
        print("wrong input")
        return 

    # train the SVM
    SVM_defined.fit(X_train, y_train)
    # test the SVM
    y_pred = SVM_defined.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))
    print("Overall accuracy for", name, round(metrics.accuracy_score(y_test, y_pred), 2))
    
    # visualise the decision boundaries
    svmPlot(X, y, SVM_defined, name) 


'''
Grid search is simply an exhaustive search in range of hyper-parameters with uniform spacing between each sampling point in the gird.
Grid search can be applied to almost all types of machine learning algorithms.
'''


def grid_search():
    # grid search for the best parameters
    g_range = 2. ** np.arange(-10, 10, step=1)
    C_range = 2. ** np.arange(-10, 10, step=1)
    parameters = [{'gamma': g_range, 'C': C_range, 'kernel': ['rbf']}]
    score = 'precision'
    # begin grid search
    grid = GridSearchCV(svm.SVC(), parameters, cv=10, n_jobs=-1)
    # n_jobs = -1 means using all processors
    grid.fit(X_train, y_train)

    print("\nGrid scores on development set\n--only mean higher than .80 will be printed\n")
    for mean, std, params in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['std_test_score'], grid.cv_results_['params']):
        if mean > 0.8:
            print("%0.3f (+/-%0.03f) for %r" % (mean, std*2, params))
    
    print("\nBest parameters set found on development set\n")
    print(grid.best_params_)

    # best hyper-parameters
    bestG = grid.best_params_['gamma']
    bestC = grid.best_params_['C']

    print("\nThe best parameters are: gramma: ", np.log2(bestG), " and Cost: ", np.log2(bestC))

    return bestG, bestC


def rbf_svm_best_para(best_g, best_C):
    # best-parameter RBF SVM
    SVM_rbf_para = svm.SVC(kernel='rbf', C=best_C, gamma=best_g)
    # train the SVM
    SVM_rbf_para.fit(X_train, y_train)
    # test the SVM
    y_pred = SVM_rbf_para.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))
    print("Overall accuracy for the best-parameter RBF SVM", round(metrics.accuracy_score(y_test, y_pred), 2), end=' ')
    print("the parameter gamma %f, C %f" % (np.log2(best_g), np.log2(best_C)))

    # visualise the decision boundaries
    svmPlot(X, y, SVM_rbf_para, 'rbf') 


if __name__ == "__main__":
    
    # linear_svm(1)
    # linear_svm(10)
    # linear_svm(100)
    # linear_svm(1000)
    # linear_svm(10000)

    # apply_one_svm('linear')
    apply_one_svm('poly')
    # apply_one_svm('rbf')

    # bestG, bestC = grid_search()
    # rbf_svm_best_para(bestG, bestC)
    
