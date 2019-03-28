import json
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from keras.utils import plot_model

config = json.load(open('./config.json', 'r'))
data_dim = config['lstm']['data_dim']
hidden = config['lstm']['hidden']
epochs = config['lstm']['epochs']
plot = config['lstm']['plot']

data = list(range(data_dim))


def reconstruction_lstm_ae():
    # define input sequence
    sequence = np.array(data)
    # reshape input into [samples, timesteps, features]
    n_in = len(sequence)
    sequence = sequence.reshape((1, n_in, 1))

    # define mode
    model = Sequential()
    model.add(LSTM(hidden, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(hidden, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(sequence, sequence, epochs=epochs, verbose=1)
    if plot:
        plot_model(model, show_shapes=True, to_file='./images/reconstruct_lstm-ae.png')

    # demonstrate reconstruction
    yhat = model.predict(sequence, verbose=1)
    print(yhat[0,:,0])


def prediction_lstm_ae():
    # define input sequence
    seq_in = np.array(data)
    # reshape input into [samples, timesteps, features]
    n_in = len(seq_in)
    seq_in = seq_in.reshape((1, n_in, 1))
    # prepare output sequence
    seq_out = seq_in[:, 1:, :]
    n_out = n_in - 1

    # define model
    model = Sequential()
    model.add(LSTM(hidden, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_out))
    model.add(LSTM(hidden, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')
    
    # fit model
    model.fit(seq_in, seq_out, epochs=epochs, verbose=1)
    if plot:
        plot_model(model, show_shapes=True, to_file='./images/prediction_lstm-ae.png')
    
    # demonstrate prediction
    yhat = model.predict(seq_in, verbose=1)
    print(yhat[0,:,0])


def composite_lstm_ae():
    # define input sequence
    seq_in = np.array(data)
    # reshape input into [samples, timesteps, features]
    n_in = len(seq_in)
    seq_in = seq_in.reshape((1, n_in, 1))
    # prepare output sequence
    seq_out = seq_in[:, 1:, :]
    n_out = n_in - 1

    # define encoder
    visible = Input(shape=(n_in, 1))
    encoder = LSTM(hidden, activation='relu')(visible)

    # define reconstruction decoder
    decoder1 = RepeatVector(n_in)(encoder)
    decoder1 = LSTM(hidden, activation='relu', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)

    # define prediction decoder
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(hidden, activation='relu', return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)

    # tie together
    model = Model(input=visible, outputs=[decoder1, decoder2])
    model.compile(optimizer='adam', loss='mse')
    if plot:
        plot_model(model, show_shapes=True, to_file='./images/composite_lstm-ae.png')

    # fit model
    model.fit(seq_in, [seq_in, seq_out], epochs=epochs, verbose=1)

    # demonstrate prediction
    yhat = model.predict(seq_in, verbose=1)
    print(yhat[0][0,:,0])
    print(yhat[1][0,:,0])


def standalone_lstm_encoder():
    # define input sequence
    sequence = np.array(data)
    # reshape input into [samples, timesteps, features]
    n_in = len(sequence)
    sequence = sequence.reshape((1, n_in, 1))

    # define model
    model = Sequential()
    model.add(LSTM(hidden, activation='relu', input_shape=(n_in, 1)))
    model.add(RepeatVector(n_in))
    model.add(LSTM(hidden, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(sequence, sequence, epochs=epochs, verbose=1)

    # connect the encoder LSTM as the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[0].output)
    if plot:
        plot_model(model, show_shapes=True, to_file='./images/lstm-encoder.png')

    # get the feature vector for the input sequence
    yhat = model.predict(sequence)
    print(yhat.shape)
    print(yhat)
    

if __name__ == "__main__":
    reconstruction_lstm_ae()
    # prediction_lstm_ae()
    # composite_lstm_ae()
    # standalone_lstm_encoder()