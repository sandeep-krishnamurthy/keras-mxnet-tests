'''
This code is forked from https://github.com/fchollet/keras/blob/master/examples/
and modified to use as MXNet-Keras integration testing for functionality and sanity performance
benchmarking.

Example script showing how to use stateful RNNs
to model long sequences efficiently.
'''
from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Imports for benchmarking
from utils.profiler import profile
from utils.model_util import make_model

# Imports for assertions
from utils.assertion_util import assert_results

#Result dictionary
global ret_dict
ret_dict = dict()


# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
batch_size = 25
# Ideal epochs is 25
epochs = 5
# number of elements ahead that are used to make the prediction
lahead = 1


def gen_cosine_amp(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
    """Generates an absolute cosine time series with the amplitude
    exponentially decreasing

    Arguments:
        amp: amplitude of the cosine function
        period: period of the cosine function
        x0: initial x of the time series
        xn: final x of the time series
        step: step of the time series discretization
        k: exponential rate
    """
    cos = np.zeros(((xn - x0) * step, 1, 1))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
        cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
    return cos


print('Generating Data')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
    expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()
model.add(LSTM(50,
               batch_input_shape=(batch_size, tsteps, 1),
               return_sequences=True,
               stateful=True))
model.add(LSTM(50,
               return_sequences=False,
               stateful=True))
model.add(Dense(1))
model = make_model(model, loss='mse', optimizer='rmsprop')

print('Training')
def train_func():
    for i in range(epochs):
        print('Epoch', i, '/', epochs)
        history = model.fit(cos,
                            expected_output,
                            batch_size=batch_size,
                            verbose=1,
                            nb_epoch=1,
                            shuffle=False)
        #ret_dict["training_accuracy"] = history.history['acc'][-1]
        #ret_dict["test_accuracy"] = history.history['val_acc'][-1]
        model.reset_states()

def test_stateful_lstm():
    ret = profile(train_func)

    ret_dict["training_time"] = str(ret[0]) + ' sec'
    ret_dict["max_memory"] = str(ret[1]) + ' MB'

    print("Test stateful lstm")
    print(ret_dict)
    # TODO: ASSERT results. Above tests whether it is functional. Assert statements will confirm accuracy/memory/speed.
