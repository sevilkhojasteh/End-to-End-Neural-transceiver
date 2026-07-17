import tensorflow as tf

class PhaseShifter(tf.keras.layers.Layer):
    def __init__(self, theta, **kwargs):
        super(PhaseShifter, self).__init__(**kwargs)
        self.theta = tf.constant(theta, dtype=tf.float32)
        
    def call(self, inputs):
        # h = cos(theta) + j*sin(theta)
        h = tf.complex(tf.cos(self.theta), tf.sin(self.theta))
        return inputs * h

# Test with 90 degrees shift (pi / 2) on a signal '1.0 + 0j'
shifter = PhaseShifter(theta=3.14159265 / 2.0)
input_signal = tf.complex([1.0], [0.0])
print("Shifted:", shifter(input_signal).numpy())
# Output is very close to [0.0 + 1.0j] (which is pure imaginary 'j')!