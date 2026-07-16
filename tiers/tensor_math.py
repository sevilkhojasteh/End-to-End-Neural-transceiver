import tensorflow as tf

# Creating a real tensor
real_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)

# Creating a complex tensor
complex_tensor = tf.complex(real_tensor, real_tensor*0.5)

# output = [[1.0 + 0.5j, 2.0 + 1.0j], [3.0 + 1.5j, 4.0 + 2.0j]]

