import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

import numpy as np
import tensorflow


"""
TO DO
-----
LOAD DATASET
EVALUATE MODEL
"""
path = ""

X_train = np.load(path + "X_train.npy")
X_test = np.load(path + "X_test.npy")
y_train = np.load(path + "y_train.npy")
y_test = np.load(path + "y_test.npy")

shape_x, shape_y = 48, 48

n_rows, n_cols, n_dims = X_train.shape[1:]
input_shape = (n_rows, n_cols, n_dims)
labels = np.unique(y_train)
n_labels = len(labels)

input_img = Input(shape=(shape_x, shape_y, 1))

# 1st layer
layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

# 2nd layer
layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)

# 3rd layer
layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)

mid_1 = tensorflow.keras.layers.concatenate([layer_1, layer_2, layer_3], axis=3)

flat_1 = Flatten()(mid_1)
dense_1 = Dense(1200, activation='relu')(flat_1)
dense_2 = Dense(600, activation='relu')(dense_1)
dense_3 = Dense(150, activation='relu')(dense_2)
output = Dense(n_labels, activation='softmax')(dense_3)

# build Inception model
model = Model([input_img], output)

# plot Inception model
plot_model(model, to_file='./images/model.png', show_shapes=True, show_layer_names=True)

# compile Inception model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 128
epochs = 150

# train Inception model
history = model.fit(X_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(X_test, y_test
))

