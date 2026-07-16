import tensorflow as tf

# Define a batch of 2 transmitted QPSK symbols: [1+1j, -1-1j]
s = tf.complex([1.0, -1.0], [1.0, -1.0]) # Shape:(2, )

# Simulate a channel phase shift of 45 degrees (pi/4 radians)
theta = tf.constant(3.14159265 / 4.0 , dtype=tf.float32)
h = tf.complex(tf.cos(theta), tf.sin(theta)) # h = cos(theta) + j*sin(theta)

# Apply phase shift: y = h * s

y_phase = h * s

# Simulate additive white gaussian Noise (AWGN)
# Complex noise has variance sigma^2/2 per real and imaginary dimension

sigma = 0.1
noise_r = tf.random.normal(shape=s.shape, mean=0.0, stddev=sigma / tf.sqrt(2.0))
noise_i = tf.random.normal(shape=s.shape, mean=0.0, stddev=sigma / tf.sqrt(2.0))

n = tf.complex(noise_r, noise_i)