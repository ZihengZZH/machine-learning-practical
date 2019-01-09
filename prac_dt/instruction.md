# Decision Tree & Random Forest coding practical

[Wisconsin Breast Cancer dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), which records clinical measurements of breast cancer tumors (569 data points and 30 features); Each tumor is labeled as benign (for harmless tumor) or malignant (for cancerous tumors), and the task is to learn to predict whether a tumor is malignant based on the measurements of the tissue.

## classifier
Decision tree classifier is capable of both binary (where the labels are [-1, 1]) classification and multiclass (where the labels are [0, ..., K-1]) classification.

## regressor
Decision tree can also be applied to regression problems, using the DecisionTreeRegressor class. As in the classification setting, the fit method will take as argument arrays X and y, only that in this case y is expected to have floating point values instead of integer values.

## predict the RAM prices
We will make a forecast for the years after 2000 using the historical data up to that point, with the date as our only feature. We will compare two simple models: a DecisionTreeRegressor and LinearRegression

![prediction_RAM]('./prediction_RAM.png')

According to the figure of predictions on RAM prices, the linear model approximates the data with a line that provides quite a good forecast for the test data (the years after 2000). The tree model, on the other hand, makes perfect predictions on the training data. However, once we leave the data range for which the model has data, the model simply keeps predicting the last known point. __The tree has no ability to generate new responses, outside of what was seen in the training data. This shortcoming applies to all models based on trees__

# Random Forests for the cancer data
As the refinement of bagges (decision) trees, Random Forests try to improve on bagging by de-correlating the trees. Typically, the feature importances provided by the random forests are more reliable than the ones provided by a single (decision) tree. As seen in the figures, the random forest gives non-zero importances to many more features than the single (decision) tree. Similar to the single decision tree, the random forest also gives a lot of importance to the worst radius feature, but it actually chooses worst perimeter to be the most informative feature overall. The randomness in the building the random forest forces the algorithm to consider many possible explanations, the result being that the random forest captures a much broader picture of the data than a single tree.

![feature-importance-dt]('./feature-importance-dt.png')
__feature-importance-dt__

![feature-importance-rf]('./feature-importance-rf.png')
__feature-importance-rf__