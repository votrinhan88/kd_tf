from typing import List

import tensorflow as tf
keras = tf.keras

class AlexNet(keras.Model):
    """ImageNet classification with deep convolutional neural networks - Krizhevsky
    et al. (2012)
    DOI: 10.1145/3065386

    Args:
        `half`: Flag to choose between AlexNet or AlexNet-Half. Defaults to `False`.
        `input_dim`: Dimension of input images. Defaults to `[32, 32, 3]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `False`.
    
    Kwargs:
        Additional keyword arguments passed to `keras.Model.__init__`.
    
    Two versions: AlexNet and AlexNet-Half following the architecture in 'Zero-Shot
    Knowledge Distillation in Deep Networks' - Nayak et al. (2019)
    Implementation: https://github.com/nphdang/FS-BBT/blob/main/cifar10/alexnet_model.py
    """    
    _name = 'AlexNet'

    def __init__(self,
                 half:bool=False,
                 input_dim:List[int]=[32, 32, 3],
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        """Initialize model.
        
        Args:
            `half`: Flag to choose between AlexNet or AlexNet-Half. Defaults to `False`.
            `input_dim`: Dimension of input images. Defaults to `[32, 32, 3]`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `False`.
        
        Kwargs:
            Additional keyword arguments passed to `keras.Model.__init__`.
        """    
        assert isinstance(half, bool), "'half' should be of type bool"
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        if half is False:
            super().__init__(name=self._name, **kwargs)
            divisor = 1
        elif half is True:
            super().__init__(name=self._name + '-Half', **kwargs)
            divisor = 2

        self.half = half
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits
        
        # Convolutional blocks
        self.conv_1 = keras.Sequential([
            keras.layers.Conv2D(filters=48//divisor, kernel_size=(5, 5), strides=(1, 1), padding='same', bias_initializer='zeros'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
            keras.layers.BatchNormalization()
        ])
        self.conv_2 = keras.Sequential([
            keras.layers.Conv2D(filters=128//divisor, kernel_size=(5, 5), strides=(1, 1), padding='same', bias_initializer='ones'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
            keras.layers.BatchNormalization()
        ])
        self.conv_3 = keras.Sequential([
            keras.layers.Conv2D(filters=192//divisor, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='zeros'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.BatchNormalization()
        ])
        self.conv_4 = keras.Sequential([
            keras.layers.Conv2D(filters=192//divisor, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='ones'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.BatchNormalization()
        ])
        self.conv_5 = keras.Sequential([
            keras.layers.Conv2D(filters=128//divisor, kernel_size=(3, 3), strides=(1, 1), padding='same', bias_initializer='ones'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
            keras.layers.BatchNormalization()
        ])

        self.flatten = keras.layers.Flatten(name='flatten')
        # Fully-connected layers
        self.fc_1 = keras.Sequential([
            keras.layers.Dense(512//divisor, bias_initializer='zeros'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization()
        ])
        self.fc_2 = keras.Sequential([
            keras.layers.Dense(256//divisor, bias_initializer='zeros'),
            keras.layers.Activation(tf.nn.relu),
            keras.layers.Dropout(0.5),
            keras.layers.BatchNormalization()
        ])
        
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits')
    
    def call(self, inputs, training:bool=False):
        x = self.conv_1(inputs, training=training)
        x = self.conv_2(x, training=training)
        x = self.conv_3(x, training=training)
        x = self.conv_4(x, training=training)
        x = self.conv_5(x, training=training)
        x = self.flatten(x)
        x = self.fc_1(x, training=training)
        x = self.fc_2(x, training=training)
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
        batch_size_train=64,
        batch_size_test=1024
    )

    net = AlexNet(
        input_dim=[28, 28, 1],
        num_classes=10,
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
        validation_data=ds['test']
    )