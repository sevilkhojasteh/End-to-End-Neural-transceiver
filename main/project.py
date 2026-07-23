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

# Shoot rays in a specific mathematically spaced pattern (Fibonacci)
# Compute paths up to 3 bounces (max_depth=3)
ray_tracer = scene.compute_paths(method="fibonacci", max_depth=3)

# Build our physical channel model from these computed ray paths
channel_model = sn.rt.PropagationModel(ray_tracer)

"""
max_depth=3: Tells the computer to track rays that bounce up to 3 times before hitting the receiver.
channel_model: This is the final output. It is a highly accurate, physical LTI system model (like we studied in Tier 2!) representing the exact real-world city environment.

"""

# The Transmitter

class NeuralTx(tf.keras.layers.Layer):
    def __init__(self, num_symbols):
        super(NeuralTx, self).__init__()
        self.num_symbols = num_symbols

        # Hidden layer to learn constellation mappings
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        # Output layer. Each complex symbol has 2 parts (Real, Imag)
        self.dense2 = tf.keras.layers.Dense(self.num_symbols * 2, activation=None)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2

        # Reshape to separate real and imaginary parts
        # New shape: [batch_size, 8 symbols, 2 parts]

        x = tf.reshape(x, [-1, self.num_symbols, 2])

        # construct the comples symbols: s = I + jQ
        symbols = tf.complex(x[..., 0], x[..., 1])

        # Calculate the average power of our generated complex symbols
        mean_power = tf.reduce_mean(tf.square(tf.abs(symbols)))

        # Normalize: divide sumbols by the square root of average power
        # This forces the average power to be exactly 1.0!
        normalized_symbols = symbols / tf.cast(tf.sqrt(mean_power), tf.complex64)
        return normalized_symbols
    

class NeuralRx(tf.keras.layers.Layer):
    def __init__(self, k):
        super(NeuralRx, self).__init__()
        self.k = k
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        # Output layer: output k nodes
        self.dense3 = tf.keras.layers.Dense(self.k, activation='sigmoid')

    def call(self, inputs):
        # 1. Split complex numbers into real and imaginary parts
        x = tf.stack([tf.real(inputs), tf.imag(inputs)], axis=-1)
        x = tf.reshape(x, [tf.shape(inputs)[0], -1]) # Flatten to 2D
        
        # 2. Pass through hidden layers and output bit probabilities
        x = self.dense1(x)
        x = self.dense2(x)
        predictions = self.dense3(x)
        return predictions
    

# Phase 5
# At each training epoch, we randomly select an SNR between 5dB (noisy) and 15dB(clean)
ebno_train = tf.random.uniform(shape=(), minval=5.0, maxval=15.0)

@tf.function # Tells TensorFlow to compile this into ultra-fast machine code
def train_step(batch_size, ebno_db, channel_model):
    # 1. Generate random binary bits (0 or 1)
    bits = sn.utils.binary_source([batch_size, k])
    
    # 2. Open our Flight Recorder (tape)
    with tf.GradientTape() as tape:
        # Step A: Neural Tx maps bits to complex symbols
        tx_symbols = tx_net(bits)
        
        # Step B: Apply Rapp Power Amplifier hardware clipping (non-linear)
        tx_symbols_distorted = sn.phy.utils.rapp_pa(tx_symbols, v_sat=1.0, p=2.0)
        
        # Step C: Pass through the 3D Ray-Traced Channel (fading)
        rx_symbols_faded = channel_model(tx_symbols_distorted)
        
        # Step D: Add AWGN Noise
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=2, coderate=1.0)
        noisy_rx_symbols = sn.phy.channel.AWGN()([rx_symbols_faded, no])
        
        # Step E: Neural Rx decodes received noisy symbols
        predictions = rx_net(noisy_rx_symbols)
        
        # Step F: Calculate Binary Cross-Entropy (BCE) Loss
        loss = bce_loss(bits, predictions)
        
    # 3. Read the Flight Recorder to calculate gradients for ALL weights in Tx & Rx
    trainable_variables = tx_net.trainable_variables + rx_net.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    
    # 4. Optimizer adjusts the weights to reduce the loss
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    
    return loss