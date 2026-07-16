import tensorflow as tf

class TrainableScalar(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        # we call super() to initialize keras's background machinery
        super(TrainableScalar, self).__init__(**kwargs)

    def build(self, input_shape):
        # We define a single trainable weight, initialized to 2.0
        self.w = self.add_weight(
            name="multiplier_weight",
            shape=(), # A empty shape () means a single scalar number
            initializer=tf.keras.initializers.Constant(2.0),
            trainable=True, # This tells TensorFlow to compute gradients for it!
            dtype=tf.float32
        )
        # We must call super().build() at the end
        super(TrainableScalar, self).build(input_shape)
        
    def call(self, inputs):
        # The math: we multiply the input by our trainable weight
        return inputs * self.w

# --- Let's test our new custom layer! ---
# 1. Instantiate the layer
scaler_layer = TrainableScalar()

# 2. Create some input data: x = [1.0, 2.0, 3.0]
x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

# 3. Pass x through the layer
y = scaler_layer(x)

print("Layer Output:", y.numpy()) # Output should be [2.0, 4.0, 6.0] because w was initialized to 2.0

print("Trainable variables inside this layer:", scaler_layer.trainable_variables)


neg_data = tf.constant([-2.0, 3.0, -1.0, 5.0])

# ReLU makes negatives 0, keeps positives
relu_output = tf.keras.activations.relu(neg_data)
print("ReLU Output   :", relu_output.numpy()) # [0. 3. 0. 5.]

# Sigmoid squashes everything to (0, 1)
sigmoid_output = tf.keras.activations.sigmoid(neg_data)
print("Sigmoid Output:", sigmoid_output.numpy())