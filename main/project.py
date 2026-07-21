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
        # Step 1: Generate random binary bits (0s and 1s)
        # Shape: [Batch_Size, K]
        bits = sn.utils.binary_source([batch_size, self.k])
        
        # Step 2: Run the Spell-Checker (Add redundancy)
        # Converts self.k bits to self.n bits
        codeword = self.encoder(bits)
        
        # Step 3: Map bits to complex wave coordinates (I/Q symbols)
        x = self.mapper(codeword)
        
        # Step 4: Calculate the noise level (No) from our Eb/No in dB
        coderate = self.k / self.n  # Percentage of useful bits
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=2, coderate=coderate)
        
        # Step 5: Add noise in the air
        y = self.channel([x, no])
        
        # Step 6: Demap received coordinates back to bit likelihoods (LLRs)
        llr = self.demapper([y, no])
        
        # Step 7: Decode LLRs back to clean bits
        decoded_bits = self.decoder(llr)
        
        return bits, decoded_bits


# Loading the 3D scene
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

# Load the built-in 3D model of Munich, Germany
scene = load_scene(sn.rt.scene.munich)

"""
The Scene: This is a 3D mesh map containing the exact coordinates and materials of buildings, streets, 
and terrain in downtown Munich. The computer knows whether a building is made of concrete or glass because different materials reflect radio waves differently!

"""

# Define a 4x4 grid of antennas (16 antennas total)
scene.tx_array = PlanarArray(
    num_rows=4, 
    num_cols=4, 
    vertical_spacing=0.5,   # Spacing between antennas (measured in wavelengths)
    horizontal_spacing=0.5, # Spacing between antennas
    pattern="dipole"        # The shape of each antenna's radiation pattern
)

# 1. Place a Transmitter (gNodeB cell tower) on a tall building roof (30 meters high)
tx = Transmitter(name="gNodeB", position=[50.0, 50.0, 30.0])

# 2. Place a Receiver (User Equipment - UE) at street level (1.5 meters high)
rx = Receiver(name="UE", position=[120.0, 80.0, 1.5])

# Add them to our Munich map
scene.add(tx)
scene.add(rx)