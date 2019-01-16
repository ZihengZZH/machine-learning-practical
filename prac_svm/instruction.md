# Support Vector Machine coding practical
In this practical, you will learn how to apply Support Vector Machine (SVM) to real-world data mining problems. We will use the SVM module from [scikit-learn](https://scikit-learn.org/stable/), a very popular machine learning package for Python. SVM has been a very popular machine learning classifier for high-dimensional dataset with relatively small sample size since its invention in '90s. In this practical session, we will first apply SVM on the iris flower dataset. You will be able to visualise the decision boundary of different classes. You will also learn to tune the hyperparameters using grid search. At the end, you will also have a chance to explore a bigger, more complex dataset of your choice.

## results and answers
refer to my SVM.ipynb

## precision is ill-defined
precision = TP / (TP + FP) = 0 if predictor does not predict positive class - TP is 0
recall = TP / (TP + FN) = 0 if predictor does not predict positive class - TP is 0

## gamma in RBF kernel
Technically, the gamma parameter is the inverse of the standard deviation of the RBF kernel (Gaussian function), which is used as similarity measure between two points. Intuitively, a small gamma value defines a Gaussian function with a large variance. In this case, two points can be considered similar even if are far from each other. On the other hand, a large gamma value means defining a Gaussian function with a small variance and in this case, two points are considered similar just if they are close to each other. 

USE GRID SEARCH TO FIND THE GAMMA 