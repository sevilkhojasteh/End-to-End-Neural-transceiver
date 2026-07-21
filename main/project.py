import tensorflow as tf
import sionna as sn

class Classical5GLink(tf.keras.Model):
    def __init__(self, k, n, num_bits_per_symbol):
        super(Classical5GLink, self).__init__()
        self.k = k # Information bits
        self.n = n # Codeword bits

        