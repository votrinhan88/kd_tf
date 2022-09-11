from typing import List
import tensorflow as tf
keras = tf.keras

class LeNet_5(keras.Model):
    '''Gradient-based learning applied to document recognition
    DOI: 10.1109/5.726791

    Two versions: LeNet-5 and LeNet-5-HALF
    '''
    _name = 'LeNet-5'

    def __init__(self, half:bool=False, input_dim:List[int]=[32, 32, 1], num_outputs:int=10, *args, **kwargs):        
        """Initialize model.

        Args:
            half (bool, optional): flag to choose between LeNet-5 or LeNet-5-HALF. Defaults to False.
            input_dim (List[int], optional): dimension of input images. Defaults to [32, 32, 1].
            num_outputs (int, optional): number of output nodes. Defaults to 10.
        """
        assert isinstance(half, bool), "'half' should be of type bool"
        
        self.half = half
        self.input_dim = input_dim
        self.num_outputs = num_outputs

        if self.half == False:
            super().__init__(name=self._name, *args, **kwargs)
            divisor = 1
        elif self.half == True:
            super().__init__(name=self._name + '-HALF', *args, **kwargs)
            divisor = 2

        self.C1      = keras.layers.Conv2D(filters=6//divisor, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C1')
        self.S2      = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid', name='S2')
        self.C3      = keras.layers.Conv2D(filters=16//divisor, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C3')
        self.S4      = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid', name='S4')
        self.C5      = keras.layers.Conv2D(filters=120//divisor, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C5')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.F6      = keras.layers.Dense(units=84//divisor, activation='tanh', name='F6')
        self.F7      = keras.layers.Dense(units=self.num_outputs, name='F7')

    def call(self, inputs):
        x = self.C1(inputs)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.F7(x)
        return x

    def build(self):
        inputs = keras.layers.Input(shape=self.input_dim)
        super().build(input_shape=[None]+self.input_dim)
        self.call(inputs)

if __name__ == '__main__':
    net = LeNet_5()
    net.build()
    net.summary()

    net = LeNet_5(half=True)
    net.build()
    net.summary()