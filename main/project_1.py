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

encoder = NeuralTransmitter(num_symbols=NUM_SYMBOLS)
decoder = NeuralReceiver(k=K)
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
bce_loss = tf.keras.losses.BinaryCrossentropy()

@tf.function
def train_step(batch_size, ebno_db):
    bits = tf.random.uniform(shape=[batch_size, K], minval=0, maxval=2, dtype=tf.int32)
    bits_float = tf.cast(bits, tf.float32)

    with tf.GradientTape as tape:
        tx_symbols = encoder(bits_float)
        tx_distorted = rapp_power_amplifier(tx_symbols, v_sat=1.0, p=2.0)
        h_real = tf.random.normal(shape=[batch_size, 1], mean=0.0, stddev=1.0 / np.sqrt(2))
        h_imag = tf.random.normal(shape=[batch_size, 1], mean=0.0, stddev=1.0 / np.sqrt(2))
        h = tf.complex(h_real, h_imag)

        faded_symbols = tx_distorted * h

        snr_linear = 10.0 ** (ebno_db / 10.0)
        r = K / NUM_SYMBOLS
        sigma = tf.sqrt(1.0 / (2.0 * r * snr_linear))

        noise_r = tf.random.normal(shape=tf.shape(faded_symbols), mean=0.0, stddev=sigma)
        noise_i = tf.random.normal(shape=tf.shape(faded_symbols), mean=0.0, stddev=sigma)
        noise = tf.complex(noise_r, noise_i)
        rx_symbols = faded_symbols + noise

        rx_equalized = rx_symbols / h
        
        # Step F: Neural Receiver decodes predictions
        predictions = decoder(rx_equalized)
        
        # Step G: Compute Loss
        loss = bce_loss(bits_float, predictions)
    
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss

print(f"\n--- Starting E2E Neural Transceiver Training ---")

for epoch in range(1, EPOCHS + 1):
    # Randomly schedule training SNR
    ebno_train = tf.random.uniform(shape=(), minval=5.0, maxval=15.0)
    loss = train_step(BATCH_SIZE, ebno_train)
    
    if epoch % 100 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d} / {EPOCHS} | Train Eb/No: {ebno_train.numpy():.2f} dB | Loss (BCE): {loss.numpy():.4f}")