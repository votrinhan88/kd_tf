from typing import List
import tensorflow as tf
keras = tf.keras

class LeNet_5(keras.Model):
    '''Gradient-based learning applied to document recognition
    DOI: 10.1109/5.726791

    Two versions: LeNet-5 and LeNet-5-HALF
    Implementation: https://datahacker.rs/lenet-5-implementation-tensorflow-2-0/
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
        
        if half == False:
            super().__init__(name=self._name, *args, **kwargs)
            divisor = 1
        elif half == True:
            super().__init__(name=self._name + '-HALF', *args, **kwargs)
            divisor = 2
        
        self.half = half
        self.input_dim = input_dim
        self.num_outputs = num_outputs

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

    def get_config(self):
        return {
            'half': self.half,
            'input_dim': self.input_dim,
            'num_outputs': self.num_outputs
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class LeNet_5_ReLU_MaxPool(keras.Model):
    '''Alternative version of LeNet-5 with ReLU activation and MaxPooling layers'''
    _name = 'LeNet-5_ReLU_MaxPool'

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
            super().__init__(name='LeNet-5-HALF_ReLU_MaxPool', *args, **kwargs)
            divisor = 2

        self.C1      = keras.layers.Conv2D(filters=6//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C1')
        self.S2      = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name='S2')
        self.C3      = keras.layers.Conv2D(filters=16//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C3')
        self.S4      = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name='S4')
        self.C5      = keras.layers.Conv2D(filters=120//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C5')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.F6      = keras.layers.Dense(units=84//divisor, activation='ReLU', name='F6')
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
        super().build(input_shape=[None]+self.input_dim)
        inputs = keras.layers.Input(shape=self.input_dim)
        self.call(inputs)


if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    import tensorflow_datasets as tfds

    # Hyperparameters
    ## Model
    IMAGE_DIM = [32, 32, 1]
    ## Training
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3
    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_TEST  = 1024

    tf.config.run_functions_eagerly(True)

    # Load data
    ds = tfds.load('mnist', as_supervised=True)
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.resize(x, IMAGE_DIM[0:2])
        x = (x - 0.1307)/0.3081
        return x, y
    ds['train'] = ds['train'].map(preprocess).shuffle(60000).batch(BATCH_SIZE_TRAIN).prefetch(1)
    ds['test'] = ds['test'].map(preprocess).batch(BATCH_SIZE_TEST).prefetch(1)

    net = LeNet_5(half=True)
    net.build()
    net.summary()
    net.compile(
        metrics=['accuracy'], 
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    best_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./logs/{net.name}_best.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    net.fit(
        ds['train'],
        batch_size=BATCH_SIZE_TRAIN,
        epochs=NUM_EPOCHS,
        # callbacks=[best_callback],
        shuffle=True,
        validation_data=ds['test']
    )