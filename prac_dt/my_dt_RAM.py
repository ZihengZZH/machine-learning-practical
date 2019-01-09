import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
We will make a forecast for the years after 2000 using the historical data up to that point, with the date as our only feature. We will compare two simple models: a DecisionTreeRegressor and LinearRegression
'''

# load the data
ram_prices = pd.read_csv("./data/ram_price.csv")

# prepare the data
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]
X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price) # normalization

# train the Decision Tree regressor and Linear regressor
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# test the Decision Tree regressor and Linear regressor
X_all = ram_prices.date[:, np.newaxis] # test on all dataset
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# get the correct answer
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

plt.semilogy(data_train.date, data_train.price, label='Training data')
plt.semilogy(data_test.date, data_test.price, label='Test data')
plt.semilogy(ram_prices.date, price_tree, label='Tree prediction')
plt.semilogy(ram_prices.date, price_lr, label='Linear prediction')
plt.legend()
plt.savefig('./prediction_RAM.png', dpi=300)
