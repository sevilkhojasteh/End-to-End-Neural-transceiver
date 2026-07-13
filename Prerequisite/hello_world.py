import tensorflow as tf
import numpy as np

# Define a simple sequential model with one dense layer
model = tf.keras.Sequential([

    # Define the input shape
    tf.keras.Input(shape=(1,)),

    # Add a Dense layer
    tf.keras.layers.Dense(units=1)
])

# Compile the model with Stochastic Gradient Descent and Mean Squared Error
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# model.predict(np.array([10.0]))

print(f"Model predicted: {model.predict(np.array([10.0]), verbose=0).item():.5f}")