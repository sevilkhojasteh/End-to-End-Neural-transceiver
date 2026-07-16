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


# a constant
speed_of_light = tf.constant(299792458.0)

# a variable
trainable_weight = tf.Variable(0.5)

# to update a variable, we use the .assign() method
trainable_weight.assign(0.6)
print("Updated Weight:", trainable_weight.numpy())


symbol = tf.complex(3.0, 4.0)

real_part = tf.math.real(symbol)
imag_part = tf.math.imag(symbol)
amplitude = tf.abs(symbol)

print("Complex Symbol:", symbol.numpy())
print("Real part (I) :", real_part.numpy())
print("Imag part (Q) :", imag_part.numpy())
print("Signal Strength:", amplitude.numpy())