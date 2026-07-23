import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

K = 16
NUM_SYMBOLS = 8

BATCH_SIZE = 1024
EPOCHS = 1000
LR = 0.001

def rapp_power_amplifier(symbols, v_sat = 1.0, p = 2.0):
    amplitude = tf.abs(symbols)
    scaling = 1.0 / tf.pow(1.0 + tf.pow(amplitude / v_sat, 2.0 * p), 1.0 / (2.0 * p))
    symbols_out = symbols * tf.cast(scaling, tf.complex64)

    return symbols_out

# Encoder layer

class NeuralTransmitter(tf.keras.layers.Layer):
    def __init__(self, num_symbols, **kwargs):
        super(NeuralTransmitter, self).__init__(**kwargs)
        self.num_symbols = num_symbols

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_symbols * 2, activation= None)
        super(NeuralTransmitter, self).build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        x = tf.reshape(x, [-1, self.num_symbols, 2])
        symbols = tf.complex(x[..., 0], x[..., 1])

        mean_power = tf.reduce_mean(tf.square(tf.abs(symbols)))
        normalized_symbols = symbols / tf.cast(tf.sqrt(mean_power), tf.complex64)
        return normalized_symbols
    
class NeuralReceiver(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super(NeuralReceiver, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.k, activation='sigmoid') # Sigmoid squashes output to (0, 1) representing probability
        super(NeuralReceiver, self).build(input_shape)

    def call(self, inputs):
        x = tf.stack([tf.math.real(inputs), tf.math.imag(inputs)], axis=-1)
        x = tf.reshape(x, [tf.shape(inputs)[0], -1])

        x = self.dense1(x)
        x = self.dense2(x)
        predictions = self.dense3(x) # Output shape: [Batch_Size, K]
        return predictions

