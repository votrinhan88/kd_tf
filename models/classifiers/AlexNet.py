from typing import List

import tensorflow as tf
keras = tf.keras

class Identity(keras.layers.Layer):
    def call(self, inputs):
        return inputs

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
        
        # Workaround to access first layer's input (cannot access sub-module of
        # subclassed models)
        self.input_layer = Identity(name='input')
        # Convolutional layers
        self.conv_1 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=48//divisor, kernel_size=(5, 5), strides=(1, 1),
                    padding='same', bias_initializer='zeros'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
                keras.layers.BatchNormalization()],
            name='conv_1'
        )
        self.conv_2 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=128//divisor, kernel_size=(5, 5), strides=(1, 1),
                    padding='same', bias_initializer='ones'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
                keras.layers.BatchNormalization()
            ],
            name='conv_2'
        )
        self.conv_3 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=192//divisor, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', bias_initializer='zeros'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.BatchNormalization()
            ],
            name='conv_3'
        )
        self.conv_4 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=192//divisor, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', bias_initializer='ones'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.BatchNormalization()
            ],
            name='conv_4'
        )
        self.conv_5 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=128//divisor, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', bias_initializer='ones'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'),
                keras.layers.BatchNormalization()
            ],
            name='conv_5'
        )

        self.flatten = keras.layers.Flatten(name='flatten')
        # Fully-connected layers
        self.fc_1 = keras.Sequential(
            layers=[
                keras.layers.Dense(512//divisor, bias_initializer='zeros'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.Dropout(0.5),
                keras.layers.BatchNormalization()
            ],
            name='fc_1'
        )
        self.fc_2 = keras.Sequential(
            layers=[
                keras.layers.Dense(256//divisor, bias_initializer='zeros'),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.Dropout(0.5),
                keras.layers.BatchNormalization()
            ],
            name='fc_2'
        )
        
        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits')
    
    def call(self, inputs, training:bool=False):
        x = self.input_layer(inputs)
        x = self.conv_1(x, training=training)
        x = self.conv_2(x, )
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.flatten(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def build(self):
        super().build(input_shape=[None, *self.input_dim])
        inputs = keras.layers.Input(shape=self.input_dim)
        self.call(inputs)

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
    repo_path = os.path.abspath(os.path.join(__file__, '../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    import tensorflow_datasets as tfds
    from models.distillers.utils import add_fmap_output

    ds = tfds.load('cifar10', as_supervised=True)
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = (x - tf.constant([[[0.4914, 0.4822, 0.4465]]]))/tf.constant([[[0.2470, 0.2435, 0.2616]]])
        return x, y
    ds['test'] = ds['test'].map(preprocess).batch(1024).prefetch(1)

    net = AlexNet()
    net.build()
    net.summary(expand_nested=True)
    net.load_weights('./pretrained/cifar10/AlexNet_8746.h5')
    net.compile(metrics=['accuracy'])
    net.evaluate(ds['test'])