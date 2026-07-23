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

# 