import tensorflow as tf

class SimpleClassifier(tf.keras.Model):
    def __init__(self):
        super(SimpleClassifier, self).__init__()

        self.hidden_layer = tf.keras.layers.Dense(units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        x = self.hidden_layer(inputs)
        predictions = self.output_layer(x)

        return predictions
    

model = SimpleClassifier()
dummy_input = tf.random.normal(shape=(2, 3))

print(model(dummy_input))