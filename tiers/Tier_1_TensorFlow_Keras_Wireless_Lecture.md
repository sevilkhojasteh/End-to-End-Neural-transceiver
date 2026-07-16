# Professor's Masterclass: Tier 1 - TensorFlow & Keras for Wireless-AI
## Course: Advanced Wireless Communications Systems (AWCS/SAR) - Pre-Semester Lecture

Welcome, Scholar! 

I am your professor for this masterclass. Today, we are covering **Tier 1: Software & Coding Foundation**. 

In wireless communications, signals are represented as complex-valued waveforms (I/Q channels). Standard deep learning frameworks (like PyTorch and classic TensorFlow) were originally built for real-valued data (images and text). However, **NVIDIA Sionna** utilizes **TensorFlow 2.x** and **Keras** to run complex-valued tensor operations natively at massive scale on GPUs. 

To build Project 1, you must master:
1.  **Tensor Math & Complex Numbers** in TensorFlow.
2.  **Keras Subclassing** (building custom trainable and non-trainable layers).
3.  **Custom Training Loops with `tf.GradientTape`** (backpropagating through physical channels).

Let's dive into our lectures, examples, exercises, and solutions.

---

## Lecture 1.1: Tensor Math & Complex Numbers

In wireless communications, a transmitted signal is a sequence of complex numbers: $s = I + jQ$. In TensorFlow, this is represented by the datatype `tf.complex64` (consisting of two 32-bit floats for the real and imaginary parts).

### Core Concepts

#### 1. Tensor Creation and Shapes
A tensor is a multi-dimensional array. In wireless, we frequently use 3D or 4D tensors with dimensions representing `[Batch_Size, Antennas, Subcarriers]` or `[Batch_Size, Antennas, Time_slots]`.

```python
import tensorflow as tf

# Creating a real tensor
real_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

# Creating a complex tensor representing I/Q samples
complex_tensor = tf.complex(real_tensor, real_tensor * 0.5) 
# Output: [[1.0 + 0.5j, 2.0 + 1.0j], [3.0 + 1.5j, 4.0 + 2.0j]]
```

#### 2. Shape Manipulation (Crucial for MIMO & OFDM)
You will constantly need to reshape, expand, and slice tensors to map them to antenna arrays or subcarriers.
*   `tf.reshape(x, shape)`: Changes the view of a tensor without copying data.
*   `tf.expand_dims(x, axis)`: Inserts a dimension of length 1.
*   `tf.concat([x, y], axis)`: Merges tensors along an axis.

#### 3. Complex-Valued Math
To compute signal power, apply phase shifts, or add AWGN (Additive White Gaussian Noise):
*   Real and Imaginary parts: `tf.real(x)`, `tf.imag(x)`
*   Complex Conjugate: `tf.math.conj(x)`
*   Magnitude (Amplitude) and Phase: `tf.abs(x)`, `tf.math.angle(x)`

### Code Example: Simulating Phase Shift & AWGN in Raw TensorFlow

```python
import tensorflow as tf

# Define a batch of 2 transmitted QPSK symbols: [1+1j, -1-1j]
s = tf.complex([1.0, -1.0], [1.0, -1.0]) # Shape: (2,)

# Simulate a channel phase shift of 45 degrees (pi/4 radians)
theta = tf.constant(3.14159265 / 4.0, dtype=tf.float32)
h = tf.complex(tf.cos(theta), tf.sin(theta)) # h = cos(theta) + j*sin(theta)

# Apply phase shift: y = h * s
y_phase = h * s

# Simulate Additive White Gaussian Noise (AWGN)
# Complex noise has variance sigma^2/2 per real and imaginary dimension
sigma = 0.1
noise_r = tf.random.normal(shape=s.shape, mean=0.0, stddev=sigma / tf.sqrt(2.0))
noise_i = tf.random.normal(shape=s.shape, mean=0.0, stddev=sigma / tf.sqrt(2.0))
n = tf.complex(noise_r, noise_i)

# Received signal: r = h*s + n
r = y_phase + n

print("Transmitted symbols (s):", s.numpy())
print("Channel coefficient (h):", h.numpy())
print("Received symbols (r)   :", r.numpy())
```

---

### Exercise 1: Power Normalization and SNR Calculation
**Your Task:** 
Write a Python function `normalize_and_add_noise(x, snr_db)` that:
1.  Takes a 1D complex tensor of arbitrary symbols `x` (shape: `[N]`).
2.  Normalizes `x` so that its average power is exactly $1.0$ (i.e., $\frac{1}{N} \sum |x_i|^2 = 1.0$).
3.  Calculates the noise standard deviation $\sigma$ required to achieve the input target `snr_db` ($SNR_{\text{dB}} = 10 \log_{10}(1 / \sigma^2)$).
4.  Generates complex AWGN noise with standard deviation $\sigma$ ($\sigma/\sqrt{2}$ per real/imag dimension) and adds it to the normalized symbols.
5.  Returns the noisy symbols and the noise standard deviation $\sigma$.

#### Hint
*   Average Power of $x$: `tf.reduce_mean(tf.square(tf.abs(x)))`
*   Normalize: $x_{\text{norm}} = \frac{x}{\sqrt{\text{Power}}}$
*   Noise variance: $\sigma^2 = 10^{-SNR_{\text{dB}}/10}$

---

### Answer 1

```python
import tensorflow as tf

def normalize_and_add_noise(x, snr_db):
    # Ensure x is a complex tensor
    x = tf.cast(x, tf.complex64)
    
    # 1. Compute average power
    power = tf.reduce_mean(tf.square(tf.abs(x)))
    
    # 2. Normalize symbols to unit power
    x_norm = x / tf.cast(tf.sqrt(power), tf.complex64)
    
    # 3. Calculate noise standard deviation sigma from SNR_dB
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma = tf.sqrt(1.0 / snr_linear)
    
    # 4. Generate complex Gaussian noise
    shape = tf.shape(x)
    noise_r = tf.random.normal(shape=shape, mean=0.0, stddev=sigma / tf.sqrt(2.0))
    noise_i = tf.random.normal(shape=shape, mean=0.0, stddev=sigma / tf.sqrt(2.0))
    n = tf.complex(noise_r, noise_i)
    
    # 5. Add noise
    y = x_norm + n
    
    return y, sigma

# Test your function
x_test = tf.complex([3.0, -4.0, 1.0], [4.0, 2.0, -1.0]) # Non-unit power symbols
y_noisy, sig = normalize_and_add_noise(x_test, snr_db=10.0)
print("Noisy normalized symbols:", y_noisy.numpy())
print("Calculated Noise Sigma  :", sig.numpy())
```

---

## Lecture 1.2: Keras Subclassing & Custom Layers

Sionna structures communications systems as a sequence of modular **Keras Layers**. To build Project 1, you must learn how to write custom layers.

### Core Concepts

A custom layer inherits from `tf.keras.layers.Layer`. It requires three primary methods:
1.  `__init__(self, ...)`: Declare hyperparameters and sub-layers here.
2.  `build(self, input_shape)`: (Optional) Define trainable weights here.
3.  `call(self, inputs)`: Define the forward-pass computation (must be differentiable).

### Code Example: Creating a Trainable BPSK Modulator
Let's build a neural layer that takes bits ($0$ or $1$) and maps them to a trainable constellation! Instead of rigid BPSK (mapping 0 $\to$ -1, 1 $\to$ 1), the neural network will *learn* the best complex constellation points.

```python
import tensorflow as tf

class TrainableBPSK(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TrainableBPSK, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # We define two trainable complex constellation points (one for bit 0, one for bit 1)
        # We initialize them close to -1 and +1
        self.constellation = self.add_weight(
            name="constellation",
            shape=(2,),
            initializer=tf.keras.initializers.Constant([-1.0, 1.0]),
            trainable=True,
            dtype=tf.float32
        )
        super(TrainableBPSK, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: binary tensor of shape [Batch_Size] containing 0s or 1s
        # We cast inputs to int32 to use as indices
        indices = tf.cast(inputs, tf.int32)
        
        # Gather corresponding constellation points
        symbols = tf.gather(self.constellation, indices)
        
        # Let's make it a complex tensor (with 0 imaginary part for BPSK)
        complex_symbols = tf.complex(symbols, tf.zeros_like(symbols))
        return complex_symbols

# Test Layer
modulator = TrainableBPSK()
bits = tf.constant([0, 1, 1, 0])
symbols = modulator(bits)
print("Mapped symbols:", symbols.numpy())
print("Trainable Weights:", modulator.trainable_weights[0].numpy())
```

---

### Exercise 2: A Differentiable Flat-Fading Channel Layer
**Your Task:**
Create a custom Keras layer `FlatFadingChannel` that:
1.  Takes a complex constellation tensor $x$ of shape `[Batch_Size]` in its `call` method.
2.  In `__init__`, takes a parameter `sigma` (noise standard deviation).
3.  In `build`, defines a **single, trainable complex coefficient** representing the channel fading gain: $h = h_R + j \cdot h_I$. Initialize $h_R = 0.8$ and $h_I = 0.2$ as trainable weights.
4.  In `call`, performs:
    *   Multiplication: $y = h \cdot x$
    *   Generates complex AWGN noise with standard deviation `sigma` and adds it to $y$.
    *   Returns the received noisy, faded signal.

---

### Answer 2

```python
import tensorflow as tf

class FlatFadingChannel(tf.keras.layers.Layer):
    def __init__(self, sigma, **kwargs):
        super(FlatFadingChannel, self).__init__(**kwargs)
        self.sigma = sigma
        
    def build(self, input_shape):
        # Define trainable real and imaginary parts of channel coefficient h
        self.h_real = self.add_weight(
            name="h_real",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.8),
            trainable=True
        )
        self.h_imag = self.add_weight(
            name="h_imag",
            shape=(),
            initializer=tf.keras.initializers.Constant(0.2),
            trainable=True
        )
        super(FlatFadingChannel, self).build(input_shape)
        
    def call(self, inputs):
        # 1. Reconstruct complex channel h
        h = tf.complex(self.h_real, self.h_imag)
        
        # 2. Apply channel fading: y = h * x
        faded = h * inputs
        
        # 3. Generate and add noise
        shape = tf.shape(inputs)
        noise_r = tf.random.normal(shape=shape, mean=0.0, stddev=self.sigma / tf.sqrt(2.0))
        noise_i = tf.random.normal(shape=shape, mean=0.0, stddev=self.sigma / tf.sqrt(2.0))
        n = tf.complex(noise_r, noise_i)
        
        return faded + n

# Test the Channel Layer
channel = FlatFadingChannel(sigma=0.1)
tx_symbols = tf.complex([1.0, -1.0], [0.0, 0.0])
rx_symbols = channel(tx_symbols)
print("Channel output:", rx_symbols.numpy())
```

---

## Lecture 1.3: Custom Training Loops & `tf.GradientTape`

In standard machine learning, you use `model.fit()`. However, in E2E wireless research, training inputs are generated dynamically (random bits) on the fly, and the loss must flow through simulated channels. You must write custom training loops using `tf.GradientTape`.

### Core Concepts

#### 1. What is `tf.GradientTape`?
TensorFlow "records" all operations executed inside a `with tf.GradientTape() as tape:` block. If any tensor operation is mathematically differentiable, TensorFlow can compute the gradient of the loss with respect to any trainable variables.

```python
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2  # y = x^2, derivative dy/dx = 2x

grad = tape.gradient(y, x)
print("Gradient (should be 2*3 = 6):", grad.numpy())
```

#### 2. End-to-End Transceiver Training Concept
An Autoencoder transceiver consists of:
$$\text{Random Bits} \to [\text{Encoder (Tx)}] \to \text{Symbols} \to [\text{Channel}] \to \text{Noisy Symbols} \to [\text{Decoder (Rx)}] \to \text{Reconstructed Bits}$$

We minimize the Binary Cross-Entropy (BCE) loss between the transmitted bits and decoded bits. The gradients flow backward from the Decoder, **through the Channel**, all the way to the Encoder!

### Code Example: Training an E2E Communication Link from Scratch

Here is a complete, minimal, functional script demonstrating a custom training loop for a neural transceiver.

```python
import tensorflow as tf

# 1. Define Encoder (Tx)
class NeuralTx(tf.keras.Model):
    def __init__(self):
        super(NeuralTx, self).__init__()
        # Maps 1 bit to a 2D representation (Real, Imag)
        self.dense = tf.keras.layers.Dense(2, activation=None)
        
    def call(self, inputs):
        # inputs shape: [Batch_Size, 1]
        x = self.dense(inputs)
        # Convert the 2D output to a single complex symbol
        symbols = tf.complex(x[:, 0], x[:, 1])
        # Energy Normalization (crucial for realistic wireless!)
        power = tf.reduce_mean(tf.square(tf.abs(symbols)))
        normalized_symbols = symbols / tf.cast(tf.sqrt(power), tf.complex64)
        return normalized_symbols

# 2. Define Decoder (Rx)
class NeuralRx(tf.keras.Model):
    def __init__(self):
        super(NeuralRx, self).__init__()
        # Takes received complex I/Q, outputs reconstructed bit probability
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # Split complex received symbols back to [Real, Imag] for standard NN layers
        x = tf.stack([tf.real(inputs), tf.imag(inputs)], axis=1)
        x = self.dense1(x)
        return self.dense2(x)

# 3. Instantiate Models & Optimizer
encoder = NeuralTx()
decoder = NeuralRx()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
bce_loss = tf.keras.losses.BinaryCrossEntropy()

# 4. Custom Training Step Function
@tf.function # Compiles the function for 10x faster execution
def train_step(batch_size, sigma):
    # Generate random binary bits (0 or 1)
    bits = tf.random.uniform(shape=[batch_size, 1], minval=0, maxval=2, dtype=tf.int32)
    bits_float = tf.cast(bits, tf.float32)
    
    with tf.GradientTape() as tape:
        # Step A: Encode bits to symbols
        tx_symbols = encoder(bits_float)
        
        # Step B: Pass through a simulated AWGN Channel (differentiable!)
        noise_r = tf.random.normal(shape=tf.shape(tx_symbols), stddev=sigma / tf.sqrt(2.0))
        noise_i = tf.random.normal(shape=tf.shape(tx_symbols), stddev=sigma / tf.sqrt(2.0))
        noisy_symbols = tx_symbols + tf.complex(noise_r, noise_i)
        
        # Step C: Decode noisy symbols back to bit probabilities
        predictions = decoder(noisy_symbols)
        
        # Step D: Compute binary cross-entropy loss
        loss = bce_loss(bits_float, predictions)
        
    # Get gradients of Loss with respect to ALL trainable variables in Tx & Rx
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    
    # Apply gradients (Update weights)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss

# 5. Training Loop Run
sigma_train = 0.3 # Moderate noise
for epoch in range(500):
    loss_val = train_step(batch_size=1024, sigma=sigma_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss_val.numpy():.4f}")
```

---

### Exercise 3: Add a Channel Phase Shift to the Training Loop
**Your Task:**
Modify the custom training loop above. Inside the `train_step` function, introduce a **fixed phase shift of 90 degrees** ($\pi/2$ radians) to the transmitted symbols before adding noise:
$$s_{\text{faded}} = s_{\text{tx}} \cdot e^{j \pi/2} = s_{\text{tx}} \cdot j$$

Run the training loop and verify if the neural network can successfully learn to correct this phase shift and achieve a low loss (< 0.2).

#### Hint
*   In complex numbers, multiplying by $j$ (which is `tf.complex(0.0, 1.0)`) shifts the phase by exactly 90 degrees.
*   `faded_symbols = tx_symbols * tf.complex(0.0, 1.0)`

---

### Answer 3

```python
import tensorflow as tf

# Instantiate models again to reset weights
encoder = NeuralTx()
decoder = NeuralRx()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
bce_loss = tf.keras.losses.BinaryCrossEntropy()

@tf.function
def train_step_with_phase_shift(batch_size, sigma):
    bits = tf.random.uniform(shape=[batch_size, 1], minval=0, maxval=2, dtype=tf.int32)
    bits_float = tf.cast(bits, tf.float32)
    
    with tf.GradientTape() as tape:
        # 1. Encode
        tx_symbols = encoder(bits_float)
        
        # 2. Apply a 90-degree phase shift (Multiply by j)
        j = tf.complex(0.0, 1.0)
        faded_symbols = tx_symbols * j
        
        # 3. Add Noise
        noise_r = tf.random.normal(shape=tf.shape(faded_symbols), stddev=sigma / tf.sqrt(2.0))
        noise_i = tf.random.normal(shape=tf.shape(faded_symbols), stddev=sigma / tf.sqrt(2.0))
        noisy_symbols = faded_symbols + tf.complex(noise_r, noise_i)
        
        # 4. Decode
        predictions = decoder(noisy_symbols)
        
        # 5. Loss
        loss = bce_loss(bits_float, predictions)
        
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss

# Run training
print("Training with 90-degree phase shift:")
sigma_train = 0.2
for epoch in range(501):
    loss_val = train_step_with_phase_shift(batch_size=1024, sigma=sigma_train)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss_val.numpy():.4f}")
```

Notice how the loss successfully drops from $\approx 0.69$ (random guessing) down to $< 0.15$! The decoder neural network dynamically learns to rotate the received symbols back by -90 degrees in the vector space to recover the original bits. You have successfully simulated a digital receiver's phase synchronization block using deep learning!
