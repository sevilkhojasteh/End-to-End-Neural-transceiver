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

