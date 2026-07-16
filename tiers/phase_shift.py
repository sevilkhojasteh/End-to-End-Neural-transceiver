import tensorflow as tf

# Define a batch of 2 transmitted QPSK symbols: [1+1j, -1-1j]
s = tf.complex([1.0, -1.0], [1.0, -1.0]) # Shape:(2, )

# Simulate a channel phase shift of 45 degrees (pi/4 radians)
theta = tf.constant(3.14159265 / 4.0 , dtype=tf.float32)
h = tf.complex(tf.cos(theta), tf.sin(theta)) # h = cos(theta) + j*sin(theta)