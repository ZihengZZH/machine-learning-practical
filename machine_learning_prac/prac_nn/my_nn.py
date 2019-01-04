import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold

from keras.datasets import fashion_mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


# load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# only keep a part of the training set (1st out of 6 folds), due to time constraints
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(X_train, y_train):
    X_train, y_train = X_train[test_index], y_train[test_index]
    break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# model hyperparameters
HIDDEN_SIZE = 128
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)

# other parameters
BATCH_SIZE = 128
L_RATE = 0.01
NUM_EPOCHS = 30

# define the model (typical sequences)
# Conv2D -> Conv2D -> MaxPool2D -> Flatten -> Dense -> Dense -> Output
model = Sequential()
# 1st conv
model.add(Conv2D(32, KERNEL_SIZE, activation='relu', input_shape=INPUT_SHAPE))
# 2nd conv
model.add(Conv2D(64, KERNEL_SIZE, activation='relu'))
# maxpooling
model.add(MaxPooling2D(POOL_SIZE))
model.add(Dropout(0.25))
# flattern
model.add(Flatten())
# 1st dense 
model.add(Dense(HIDDEN_SIZE))
model.add(Activation('relu'))
# 2nd dense
model.add(Dense(HIDDEN_SIZE))
model.add(Activation('relu'))
model.add(Dropout(0.25))
# output
model.add(Dense(NUM_CLASSES, activation='softmax'))

print(model.summary())

opt = SGD(lr=L_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the model and evaluate performance on unseen test data
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS) # wrong
eval_results = model.evaluate(X_test, y_test, batch_size=16)
print("Test accuracy:", eval_results[1])

# optimisation             epoch   accuracy
# sigmoid                   30      84.74%
# relu                      30      86.75%
# dropout                   30      87.68%
# dropout                   40      84.73%
# dropout + norm            40      83.46%
# conv/fc+batch+relu+drop   40      72.09%
# fc+relu+drop+batch        40      84.66%
# fc+relu+batchnorm         40      86.32%
# conv/fc+relu+batchnorm    40      83.4%
# conv&dropout+fc&batchnorm 40      84.74%
# dropout + PReLu           40      84.87%
# dropout + relu            40      85.82%
# dropout                   40      86.63%