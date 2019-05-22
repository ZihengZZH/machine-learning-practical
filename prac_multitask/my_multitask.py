import os
import sys

import numpy as np
import pandas as pd
import h5py
import keras.backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

MISSING_LABEL_PROBS = [0.75, 0.50, 0.25, 0.00]
CLASSES = np.array(['desert', 'mountain', 'sea', 'sunset', 'trees'])

num_classes = len(CLASSES)

# input image dimensions
img_rows, img_cols = 100, 100
channels = 3

BASE_DIR = '.'
DATA_DIR = 'data'

def load(test_size=.2, random_state=100):
    f = h5py.File(os.path.join(DATA_DIR, 'dataset.h5'))
    x = f['x'].value
    y = f['y'].value
    f.close()
    x_train , x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    x_train = np.rollaxis(x_train, 1, 4)
    x_test = np.rollaxis(x_test, 1, 4)
    x_train = x_train  / 255.0
    x_test = x_test / 255.0
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train_orig, y_test = load()


MISSING_LABEL_FLAG = -1


def build_masked_loss(loss_function=K.binary_crossentropy, mask_value=MISSING_LABEL_FLAG):
    """Builds a loss function that masks based on targets
    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets
    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        dtype = K.floatx()
        mask = K.cast(K.not_equal(y_true, mask_value), dtype)
        return loss_function(y_true * mask, y_pred * mask)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.cast(K.sum(K.cast(K.not_equal(y_true, MISSING_LABEL_FLAG), dtype)), dtype)
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype)) #  - K.cast(K.sum(K.cast(K.equal(y_true, MISSING_LABEL_FLAG), dtype)), dtype)
    return correct / total


def compile_model(dropouts=[.25, .25, .5],
                    num_neurons=[32, 32, 64, 64, 512, num_classes],
                    activations=['relu'] * 5 + ['sigmoid'] ):
    model = Sequential()
    model.add(Conv2D(num_neurons[0], kernel_size=(3, 3),
        padding='same', input_shape=(img_rows, img_cols, channels)))
    model.add(Activation(activations[0]))
    model.add(Conv2D(num_neurons[1], (3, 3)))
    model.add(Activation(activations[1]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropouts[0]))

    model.add(Conv2D(num_neurons[2],(3, 3), padding='same'))
    model.add(Activation(activations[2]))
    model.add(Conv2D(num_neurons[3], (3, 3)))
    model.add(Activation(activations[3]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropouts[1]))

    model.add(Flatten())
    model.add(Dense(num_neurons[-2]))
    model.add(Activation(activations[-2]))
    model.add(Dropout(dropouts[-1]))
    model.add(Dense(num_neurons[-1]))
    model.add(Activation(activations[-1]))

    model.compile(loss=build_masked_loss(),
                optimizer='adam',
                metrics=[masked_accuracy])
    return model


def infer(input_data, model):
    labels = []
    y_pred = model.predict(input_data)
    
    # Performing masking
    y_pred = (y_pred > 0.5) * 1.0
    
    for i in range(y_pred.shape[0]):
        # select the indices
        indices = np.where(y_pred[i] == 1.0)[0]
        # Adding the results 
        labels.append(CLASSES[indices].tolist())
        
    return labels


def run_trade_study(num_epochs=10, batch_size=50, missing_label_probabilities=MISSING_LABEL_PROBS):
    confusions = []
    testset_preds = []

    for missing_label_prob in missing_label_probabilities:
        print(f'Setting {int(missing_label_prob * 100)}% of the labels to {MISSING_LABEL_FLAG} (flag them as missing).')
        y_train = y_train_orig.copy()
        mask_labels_to_remove = np.random.rand(*y_train.shape) < missing_label_prob
        y_train[mask_labels_to_remove] = MISSING_LABEL_FLAG

        model = compile_model()

        # reference
        # https://github.com/tensorflow/tensorflow/issues/24828

        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                verbose=1,
                validation_data=(x_test, y_test))

        infer(x_test, model=model)
        df_pred = pd.DataFrame(model.predict(x_test), columns=['pred_' + c for c in CLASSES])
        df_true = pd.DataFrame(y_test, columns=['true_' + c for c in CLASSES])
        df = pd.concat([df_pred, df_true], axis=1)
        print(df.head())
        testset_preds.append(df)
        confusions.append([])
        for c in CLASSES:
            labels = (f'not_{c}', f'{c}')
            confusion = pd.DataFrame(
                confusion_matrix(df['true_' + c].round(), df['pred_' + c].round()),
                columns=[f'pred_{labels[0]}', f'pred_{labels[1]}'],
                index=[f'true_{labels[0]}', f'true_{labels[1]}'])
            confusions[-1].append(confusion)
            print(confusion)
            print(classification_report(df['true_' + c].round(), df['pred_' + c].round(), target_names=labels))
        label_acc = 1.0 - np.sum(np.abs((df_pred.values - df_true.values)), axis=0) / len(df_true)
        name = '_'.join([f'{label}{int(acc*100):02}' for (label, acc) in zip(CLASSES, label_acc)])
        filename = f"{int(missing_label_prob * 100):02}pct-missing-labels_{name}"

        filepath = os.path.join(DATA_DIR, filename)
        print(f"filepath: {filepath}")
        model.save(filepath + ".h5")
        df.to_csv(filepath + ".csv")
    
    # should save results in one big h5:
    # print(confusions)
    # print([df.head() for df in testset_preds])
    return testset_preds, confusions


if __name__ == '__main__':
    num_epochs = 10
    if len(sys.argv[1:]):
        num_epochs = int(sys.argv[1])
    batch_size = 50
    if len(sys.argv[2:]):
        batch_size = int(sys.argv[2])
    print(f'Starting {num_epochs} epochs of training, with batch_size={batch_size}')
    run_trade_study(num_epochs=num_epochs, batch_size=batch_size)
