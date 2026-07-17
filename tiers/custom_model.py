import tensorflow as tf

class NeuralReceiver(tf.keras.Model):
    def __init__(self):
        super(NeuralReceiver, self).__init__()
        # We declare our layers
        # Dense is keras's built-in layer that does: Output = Input * Weight + Bias
        self.hidden_layer = tf.keras.layers.Dense(units=8, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        # We cahin the layers together
        # Input shape: [Batch_size, 2] -> representing [real, imag] parts of I/Q
        x = self.hidden_layer(inputs)
        predictions = self.output_layer(x)

        return predictions
    

# Let's test our model!
receiver = NeuralReceiver()

# Imagine we receive 3 symbols. Each symbol is represented by 2 numbers: [Real, Imag]
received_signals = tf.constant([
    [ 0.9,  0.1], # Likely a "+1" symbol
    [-0.8, -0.2], # Likely a "-1" symbol
    [ 0.1,  0.8]  # Noisy symbol
], dtype=tf.float32)

# Get predictions (probilities of being bit 1)
predictions = receiver(received_signals)
print("\nReceiver Model Predictions (Probabilities):\n", predictions.numpy())