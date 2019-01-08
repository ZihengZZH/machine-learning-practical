# Decision Tree coding practical
[Wisconsin Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), which records clinical measurements of breast cancer tumors (569 data points and 30 features); Each tumor is labeled as benign (for harmless tumor) or malignant (for cancerous tumors), and the task is to learn to predict whether a tumor is malignant based on the measurements of the tissue.

## classifier
Decision tree classifier is capable of both binary (where the labels are [-1, 1]) classification and multiclass (where the labels are [0, ..., K-1]) classification.

## regressor
Decision tree can also be applied to regression problems, using the DecisionTreeRegressor class. As in the classification setting, the fit method will take as argument arrays X and y, only that in this case y is expected to have floating point values instead of integer values.
