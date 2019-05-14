from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Reshape, UpSampling2D, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.datasets import mnist
import numpy as np
import sys


def model_flat():
    input_img = Input(shape=(784,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    model = Model(input_img, decoded)
    return model


def model_conv():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # (4, 4, 8) i.e. 128-dimension

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, decoded)
    return model


def train_model(model):
    (X_train, _), (X_test, _) = mnist.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

    from keras.callbacks import TensorBoard

    model.fit(X_train, X_train, 
                validation_data=(X_test, X_test),
                epochs=30,
                batch_size=128,
                shuffle=True,
                verbose=2,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    visualize_model(model, X_test)


def visualize_model(model, X_test):
    decoded_imgs = model.predict(X_test)
    import matplotlib.pyplot as plt
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def main(choice='flat'):
    model = model_flat() if choice == 'flat' else model_conv()
    print(model.summary())
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    train_model(model)


if __name__ == "__main__":
    main(choice='conv')