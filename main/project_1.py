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


def evaluate_neural_transceiver(ebno_db_range, use_rapp_pa=True):
    ber_list = []
    # Evaluate over 10,000 parallel blocks for extreme precision
    eval_batch_size = 10000 

    for ebno in ebno_db_range:
        # Generate random evaluation bits
        bits = tf.random.uniform(shape=[eval_batch_size, K], minval=0, maxval=2, dtype=tf.int32)
        bits_float = tf.cast(bits, tf.float32)

        tx_symbols = encoder(bits_float)

        if use_rapp_pa:
            tx_symbols = rapp_power_amplifier(tx_symbols, v_sat=1.0, p=2.0)

        h_real = tf.random.normal(shape=[eval_batch_size, 1], mean=0.0, stddev=1.0 / np.sqrt(2))
        h_imag = tf.random.normal(shape=[eval_batch_size, 1], mean=0.0, stddev=1.0 / np.sqrt(2))
        h = tf.complex(h_real, h_imag)
        faded_symbols = tx_symbols * h 

        snr_linear = 10.0 ** (ebno / 10.0)
        r = K / NUM_SYMBOLS
        sigma = tf.sqrt(1.0 / (2.0 * r * snr_linear))
        noise_r = tf.random.normal(shape=tf.shape(faded_symbols), stddev=sigma)
        noise_i = tf.random.normal(shape=tf.shape(faded_symbols), stddev=sigma)
        rx_symbols = faded_symbols + tf.complex(noise_r, noise_i)

        rx_equalized = rx_symbols / h

        predictions = decoder(rx_equalized)
        decoded_bits = tf.cast(predictions >= 0.5, tf.int32)

        total_bits = eval_batch_size * K
        bit_errors = tf.reduce_sum(tf.cast(bits != decoded_bits, tf.int32))
        ber = bit_errors / total_bits
        ber_list.append(ber.numpy())

    return ber_list


def evaluate_classical_baseline(ebno_db_range, use_rapp_pa=True):
    ber_list = []
    eval_batch_size = 10000

    constellation = np.array([-1.0-1.0j, -1.0+1.0j, 1.0-1.0j, 1.0+1.0j]) / np.sqrt(2.0)
    
    for ebno in ebno_db_range:
        # Generate random bits (2 bits per symbol, so K = 16 bits = 8 QPSK symbols)
        bits = np.random.randint(0, 2, size=(eval_batch_size, K))

        grouped_bits = bits.reshape(eval_batch_size, NUM_SYMBOLS, 2)
        indices = grouped_bits[..., 0] * 2 + grouped_bits[..., 1]

        tx_symbols = constellation[indices]

        if use_rapp_pa:
            # Apply Rapp PA scaling
            amplitude = np.abs(tx_symbols)
            scaling = 1.0 / np.power(1.0 + np.power(amplitude / 1.0, 4.0), 1.0 / 4.0)
            tx_symbols = tx_symbols * scaling
