import keras
import numpy as np
from sklearn.model_selection import StratifiedKFold

from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import SGD


# Load Fashion-MNIST
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Only keep a part of the training set (1st out of 6 folds), due to time constraints
skf = StratifiedKFold(n_splits=6, random_state=0, shuffle=False)
for train_index, test_index in skf.split(X_train, y_train):
    X_train, y_train = X_train[test_index], y_train[test_index]
    break
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# Model (hyper)parameters
HIDDEN_SIZE = 128
INPUT_SHAPE = (28, 28)
NUM_CLASSES = 10

# Other hyperparameters
BATCH_SIZE = 128
L_RATE = 0.01
NUM_EPOCHS = 40


# Define the model
inp = Input(shape=INPUT_SHAPE)
flatten = Flatten()(inp) # flatten the images
fc_1 = Dense(HIDDEN_SIZE, activation='sigmoid')(flatten) # hidden layers
fc_2 = Dense(HIDDEN_SIZE, activation='sigmoid')(fc_1)
out = Dense(NUM_CLASSES, activation='softmax')(fc_2) # classification layer

model = Model(inputs=inp, outputs=out)
print(model.summary())


# Define optimization procedure
opt = SGD(lr=L_RATE)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# Train the model and evaluate performance on unseen test data
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
eval_results = model.evaluate(X_test, y_test, batch_size=16)
print("Test accuracy:", eval_results[1])
