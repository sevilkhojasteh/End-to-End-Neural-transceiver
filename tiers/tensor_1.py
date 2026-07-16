import tensorflow as tf

vector = tf.constant([1.0, 2.0, 3.0, 4.0])

result_vector = vector * 2.0 + 5.0
print("Result is: ", result_vector)

x = tf.constant(5.0)

w = tf.Variable(1.0)
with tf.GradientTape() as tape:
    y = w * x

gradient = tape.gradient(y, w)

print("The gradient is: ", gradient)