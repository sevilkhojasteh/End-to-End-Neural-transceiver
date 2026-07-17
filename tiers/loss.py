import tensorflow as tf

loss_function = tf.keras.losses.BinaryCrossentropy()

true_bit = [1.0]
good_guess = [0.95]
bad_guess = [0.05]

print("Loss for a good guess:", loss_function(true_bit, good_guess).numpy()) # Output: ~0.05
print("Loss for a bad guess :", loss_function(true_bit, bad_guess).numpy())  # Output: ~2.99


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)