import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime 
import json
import os

from keras.layers import Input, Dense, Lambda
from keras.models import Model, load_model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist


class Timer():

    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = datetime.datetime.now()

    def stop(self):
        self.end_time = datetime.datetime.now()
        print("Time taken: %s" % (self.end_time - self.start_time))


class AutoEncoder():

    def __init__(self):
        self.config = json.load(open('config.json', 'r'))
        self.batch_size = self.config['model']['batch_size']
        self.original_dim = self.config['model']['original_dim']
        self.latent_dim = self.config['model']['latent_dim']
        self.intermediate_dim = self.config['model']['intermediate_dim']
        self.epochs = self.config['model']['epochs']
        self.epsilon_std = self.config['model']['epsilon_std']
        self.save_dir = self.config['model']['save_dir']

        self.x = Input(shape=(self.original_dim, ))
        self.h = Dense(self.intermediate_dim, activation='relu')(self.x)
        self.z_mean = Dense(self.latent_dim)(self.h)
        self.z_log_var = Dense(self.latent_dim)(self.h)
        self.decoder_h = None
        self.decoder_mean = None
        self.h_decoded = None
        self.x_decoded_mean = None
        self.vae = Model()

    def loda_model(self):
        weights_list = [f for f in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, f))]
        print(weights_list)

        print("Here are the weights of pre-trained models")
        for idx, name in enumerate(weights_list):
            print("no. %d model with name %s" % (idx, name))

        choose_model = None
        while True:
            try:
                choose_model = input("Please make your choice\t")
                weights_name = weights_list[int(choose_model)]
                self.vae.load_weights(os.path.join(self.save_dir, weights_name))
                break
            except:
                print("wrong input! Please start over")

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0, stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2)* epsilon

    def build_model(self):
        z = Lambda(self.sampling, output_shape=(self.latent_dim, ))([self.z_mean, self.z_log_var])
        # instantiate these layers separately so as to reuse later
        self.decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        self.h_decoded = self.decoder_h(z)
        self.x_decoded_mean = self.decoder_mean(self.h_decoded)
        # instaniate VAE model
        self.vae = Model(self.x, self.x_decoded_mean)
        # compute VAE loss
        xent_loss = self.original_dim * metrics.binary_crossentropy(self.x, self.x_decoded_mean)
        kl_loss = -0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=1)
        vae_loss = K.mean(xent_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()

    def train_model(self, X_train, X_test):

        X_train = X_train.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

        self.vae.fit(X_train,
                    shuffle=True,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(X_test, None))

        # save the model
        save_frame = os.path.join(self.save_dir, '%s-e%s.h5' % (datetime.datetime.now().strftime('%d%m%Y-%H%M%S'), str(self.epochs)))
        self.vae.save_weights(save_frame)

    def show_prediction_latent(self, X_test, y_test):

        X_test = X_test.astype('float32') / 255.
        X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
        # build a model to predict inputs on the latent space
        encoder = Model(self.x, self.z_mean)

        X_test_encoded = encoder.predict(X_test, batch_size=self.batch_size)
        plt.figure(figsize=(6,6))
        plt.scatter(X_test_encoded[:, 0], X_test_encoded[:, 1], c=y_test)
        plt.colorbar()
        plt.show()
    
    def show_learned_distribution(self):
        # build a digit generator that can sample from the learn distribution
        decoder_input = Input(shape=(self.latent_dim, ))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        # display a 2D manifold of the digits
        n = 15
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # linearly spaced coordinates on the unit square were transformed through the inverse CDF of the Gaussian to produce values of the latent variable z, since the prior of the latent space is Gaussian
        grid_x = norm.ppf(np.linspace(.05, .95, n))
        grid_y = norm.ppf(np.linspace(.05, .95, n))

        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded = generator.predict(z_sample)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i * digit_size: (i+1) * digit_size,
                        j * digit_size: (j+1) * digit_size] = digit
        
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()


def main(use_trained=False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    my_autoencoder = AutoEncoder()
    my_autoencoder.build_model()
    if not use_trained:
        my_autoencoder.train_model(X_train, X_test)
    else:
        my_autoencoder.loda_model()
    my_autoencoder.show_prediction_latent(X_test, y_test)
    my_autoencoder.show_learned_distribution()


if __name__ == "__main__":
    main(use_trained=True)