from typing import List
import tensorflow as tf
keras = tf.keras

class LeNet_5(keras.Model):
    '''Gradient-based learning applied to document recognition
    DOI: 10.1109/5.726791

    Args:
        `half`: Flag to choose between LeNet-5 or LeNet-5-HALF. Defaults to `False`.
        `input_dim`: Dimension of input images. Defaults to `[32, 32, 1]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `False`.

    Kwargs:
        Additional keyword arguments passed to `keras.Model.__init__`.

    Two versions: LeNet-5 and LeNet-5-HALF
    Implementation: https://datahacker.rs/lenet-5-implementation-tensorflow-2-0/
    '''
    _name = 'LeNet-5'

    def __init__(self,
                 half:bool=False,
                 input_dim:List[int]=[32, 32, 1],
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        """Initialize model.
        
        Args:
            `half`: Flag to choose between LeNet-5 or LeNet-5-HALF. Defaults to `False`.
            `input_dim`: Dimension of input images. Defaults to `[32, 32, 1]`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `False`.

        Kwargs:
            Additional keyword arguments passed to `keras.Model.__init__`.
        """
        assert isinstance(half, bool), '`half` must be of type bool'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        if half is False:
            super().__init__(name=self._name, **kwargs)
            divisor = 1
        elif half is True:
            super().__init__(name=self._name + '-HALF', **kwargs)
            divisor = 2
        
        self.half = half
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.C1      = keras.layers.Conv2D(filters=6//divisor, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C1')
        self.S2      = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid', name='S2')
        self.C3      = keras.layers.Conv2D(filters=16//divisor, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C3')
        self.S4      = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='valid', name='S4')
        self.C5      = keras.layers.Conv2D(filters=120//divisor, kernel_size=5, strides=1, activation='tanh', padding='valid', name='C5')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.F6      = keras.layers.Dense(units=84//divisor, activation='tanh', name='F6')
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits')

    def call(self, inputs):
        x = self.C1(inputs)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.flatten(x)
        x = self.F6(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def build(self):
        super().build(input_shape=[None, *self.input_dim])

    def summary(self, as_functional:bool=False, **kwargs):
        """Prints a string summary of the network.

        Args:
            `as_functional`: Flag to print from a dummy functional model.
                Defaults to `False`.

        Kwargs:
            Additional keyword arguments passed to `keras.Model.summary`.
        """
        inputs = keras.layers.Input(shape=self.input_dim)
        outputs = self.call(inputs)

        if as_functional is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'half': self.half,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'return_logits': self.return_logits
        })
        return config

class LeNet_5_ReLU_MaxPool(LeNet_5):
    """Alternative version of LeNet-5 with ReLU activation and MaxPooling layers.
    """
    _name = 'LeNet-5_ReLU_MaxPool'

    def __init__(self,
                 half:bool=False,
                 input_dim:List[int]=[32, 32, 1],
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        assert isinstance(half, bool), '`half` should be of type bool'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        if half is False:
            keras.Model.__init__(self, name=self._name, **kwargs)
            divisor = 1
        elif half is True:
            keras.Model.__init__(self, name='LeNet-5-HALF_ReLU_MaxPool', **kwargs)
            divisor = 2
        
        self.half = half
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.C1      = keras.layers.Conv2D(filters=6//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C1')
        self.S2      = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name='S2')
        self.C3      = keras.layers.Conv2D(filters=16//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C3')
        self.S4      = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name='S4')
        self.C5      = keras.layers.Conv2D(filters=120//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C5')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.F6      = keras.layers.Dense(units=84//divisor, activation='ReLU', name='F6')
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits')

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)
    
    from dataloader import dataloader

    ds = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        resize=[32, 32],
        batch_size_train=64,
        batch_size_test=1024
    )

    net = LeNet_5(
        input_dim=[32, 32, 1],
        num_classes=10
    )
    net.build()
    net.summary(as_functional=True, expand_nested=True, line_length=120)
    net.compile(
        metrics=['accuracy'], 
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy())

    best_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./logs/{net.name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
    )
    csv_logger = keras.callbacks.CSVLogger(
        filename=f'./logs/{net.name}.csv',
        append=True
    )

    net.fit(
        ds['train'],
        epochs=10,
        callbacks=[best_callback, csv_logger],
        shuffle=True,
        validation_data=ds['test']
    )