import tensorflow as tf
keras = tf.keras

class HintonNet(keras.Model):
    """Baseline model in implemented in paper 'Distilling the Knowledge in a Neural
    Network' - Hinton et al. (2015)
    DOI: 10.48550/arXiv.1503.02531

    Consist of two hidden fully-connected layers.
    - Teacher: 1200 nodes in each hidden layer
    - Student: 800 nodes in each hidden layer
    """    
    _name = 'HintonNet'

    def __init__(self,
                 num_inputs:int=784,
                 num_hiddens:int=1200,
                 num_outputs:int=10,
                 *args, **kwargs):
        """Initialize model.

        Args:
            num_inputs (int, optional): number of input nodes. Defaults to 784.
            num_hiddens (int, optional): number of nodes in each hidden layer. Defaults to 1200.
            num_outputs (int, optional): number of output nodes. Defaults to 10.
        """        
        super().__init__(self, name=self._name, *args, **kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_outputs = num_outputs

        self.dense_1      = keras.layers.Dense(units=self.num_hiddens, name='dense_1')
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.3, name='leaky_relu_1')
        self.dense_2      = keras.layers.Dense(units=self.num_hiddens, name='dense_2')
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.3, name='leaky_relu_2')
        self.dense_3      = keras.layers.Dense(units=self.num_outputs, name='dense_3')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dense_3(x)
        return x

    def build(self):
        inputs = keras.layers.Input(shape=self.num_inputs)
        super().build(input_shape=[None, self.num_inputs])
        self.call(inputs)

if __name__ == '__main__':
    hnet_large = HintonNet(num_inputs=784, num_hiddens=1200, num_outputs=10)
    hnet_large.build()
    hnet_large.summary()

    hnet_small = HintonNet(num_inputs=784, num_hiddens=800, num_outputs=10)
    hnet_small.build()
    hnet_small.summary()