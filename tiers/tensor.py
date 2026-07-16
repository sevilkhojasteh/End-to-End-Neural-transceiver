import tensorflow as tf

# create a 0D tensor (scalar)
scalar = tf.constant(0.5)

# create 1D tensor (vector)
vector = tf.constant([1.0, 0.0, 1.0, 1.0])

# create 2D tensor (matrix)
matrix = tf.constant([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("our scalar:", scalar)
print("our vector:", vector)
print("our matrix:", matrix)