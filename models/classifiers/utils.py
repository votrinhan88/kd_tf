import tensorflow as tf
keras = tf.keras

class Identity(keras.layers.Layer):
    def call(self, inputs):
        return inputs
    
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)