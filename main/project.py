import tensorflow as tf
# This is the "AI brain", it handles high-speed matrix math and calculates gradients automatically
import sionna as sn
# This is a specialized library built on top os Tensorflow. It contains pre-written, highly optimized blocks representing standard cellular communication hardware (like 5g encoders, modulators, and channels)
import numpy as np


"""
Imagine you invented a brand new running shoe. How do you prove it actually makes people run faster?

You don't just measure a runner's speed with your shoe.
First, you measure their speed wearing standard, ordinary running shoes. This is your Baseline.
Then, you measure their speed wearing your new shoe, and compare the two.
In Project 1, we want to prove that our AI-Native Neural Transceiver can outperform standard cellular technology under distorted hardware conditions.
To do this, we must first build the standard, classical cellular technology. This is Phase 2: The Classical 5G NR Baseline Link.


"""
class Classical5GLink(tf.keras.Model):
    def __init__(self, k, n, num_bits_per_symbol):
        super(Classical5GLink, self).__init__()
        self.k = k # Information bits
        self.n = n # Codeword bits

        # 1. We declare pur 5G LDPC spell-checker (Encoder & Decoder)
        self.encoder = sn.phy.fec.ldpc.encoding.LDPC5GEncoder(k, n)
        # The decoder needs to know the encoder's structure to undo it
        self.decoder = sn.phy.fec.ldpc.decoding.LDPC5GDecoder(self.encoder, hard_decidions=True)

        # 2. We declare our QAM constellation Map (Mapper & Demapper)
        # num_bits_per_symbol represents how many bits we pack into 1 complex number.
        # e.g., 2 bits = QPSK (4 houses), 4 bits = 16-QAM (16 houses)
        self.mapper = sn.phy.mapping.Mapper("qam", num_bits_per_symbol)
        self.demapper = sn.phy.mapping.Demapper("app", "qam", num_bits_per_symbol)
        
        # 3. We declare our physical air channel (add noise)
        self.channel = sn.phy.channel.AWGN()

    def call(self, batch_size, ebno_db):
        

