from typing import Literal, List

import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.classifiers.utils import Identity
else:
    from .utils import Identity

# Equivalent hyperparameters for BatchNorm layer from PyTorch
EPSILON = 1e-5
MOMENTUM = 0.9

class ResidualBasicBlock_DAFL(keras.Model):
    def __init__(self,
                 filters:int=64,
                 strides:int=1,
                 first_block:bool=False,
                 activation="relu",
                 *args, **kwargs
                 ) -> keras.layers.Layer:

        super().__init__(*args, **kwargs)
        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.first_block = first_block # Unused

        self.activation_layer = keras.activations.get(activation)
        self.l2_regu = keras.regularizers.L2(l2=5e-4)

        self.main_layers = keras.Sequential([
            keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False, kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same', use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
        ])

        if self.strides == 1:
            self.skip_layers = keras.Sequential([Identity()])
        elif self.strides > 1:
            self.skip_layers = keras.Sequential([
                keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=self.strides, padding='same', use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
                keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
            ])

    def call(self, inputs):
        main_x = self.main_layers(inputs)
        skip_x = self.skip_layers(inputs)
        x = keras.layers.Activation(self.activation)(main_x + skip_x)
        return x

class ResidualBottleneck_DAFL(keras.Model):
    def __init__(self,
                 filters:int=64,
                 strides:int=1,
                 first_block:bool=False,
                 activation="relu",
                 *args, **kwargs
                 ) -> keras.layers.Layer:
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.strides = strides
        self.first_block = first_block
        self.activation = activation
        self.l2_regu = keras.regularizers.L2(l2=5e-4)

        self.main_layers = keras.Sequential([
            keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='valid', use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters=4*self.filters, kernel_size=1, strides=1, padding='valid', use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
        ])

        if (self.strides == 1) & (self.first_block is False):
            self.skip_layers = Identity()
        elif (self.strides > 1) | (self.first_block is True):
            self.skip_layers = keras.Sequential([
                keras.layers.Conv2D(filters=4*self.filters, kernel_size=1, strides=self.strides, padding='valid', use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
                keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
            ])

    def call(self, inputs):
        main_x = self.main_layers(inputs)
        skip_x = self.skip_layers(inputs)
        x = keras.layers.Activation(self.activation)(main_x + skip_x)
        return x

class ResidualLayer_DAFL(keras.Model):
    def __init__(self,
                 filters:int,
                 block_type:Literal['small', 'large'],
                 num_units:int,
                 first_block:bool=False,
                 activation="relu",
                 *args, **kwargs
                 ) -> keras.Model:
        assert isinstance(first_block, bool), '`first_block` must be of type bool.'
        assert block_type in ['small', 'large'], "`block_type` must be one of 'small', 'large'."

        super().__init__(*args, **kwargs)
        self.block_type = block_type
        self.filters = filters
        self.num_units = num_units
        self.first_block = first_block
        self.activation = activation

        res_unit_params = {
            'filters': self.filters,
            'activation': self.activation
        }

        if self.block_type == 'small':
            Block = ResidualBasicBlock_DAFL
        elif self.block_type == 'large':
            Block = ResidualBottleneck_DAFL

        self.blocks = []
        if self.first_block is True:
            self.blocks.append(Block(strides=1, first_block=self.first_block, **res_unit_params))
        elif self.first_block is False:
            self.blocks.append(Block(strides=2, first_block=self.first_block, **res_unit_params))
        for unit in range(self.num_units - 1):
            self.blocks.append(Block(strides=1, **res_unit_params))
        self.blocks = keras.Sequential(self.blocks)
        
    def call(self, inputs):
        x = self.blocks(inputs)
        return x

class ResNet_DAFL(keras.Model):
    _name = 'ResNet-DAFL'
    block_type = {
        18:  'small',
        34:  'small',
        50:  'large',
        101: 'large',
        152: 'large',
    }
    num_units = {
        18:  [2, 2, 2, 2],
        34:  [3, 4, 6, 3],
        50:  [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self,
                 ver:Literal[18, 34, 50, 101, 152]=18,
                 input_dim:List[int]=[32, 32, 3],
                 num_classes:int=10,
                 return_logits:bool=False,
                 *args, **kwargs):
        assert ver in [18, 34, 50, 101, 152], f'`ver` must be of [18, 34, 50, 101, 152].'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super().__init__(name=f'{self._name}-{ver}', *args, **kwargs)
        self.ver = ver
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits
        self.l2_regu = keras.regularizers.L2(l2=5e-4)

        # Workaround to access first layer's input (cannot access sub-module of
        # subclassed models)
        self.conv_1 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", use_bias=False,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu),
                keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
                keras.layers.Activation(tf.nn.relu),
            ],
            name='conv_1')

        self.conv_2 = ResidualLayer_DAFL(filters=64, block_type=self.block_type[ver], num_units=self.num_units[ver][0], name='conv_2', first_block=True)
        self.conv_3 = ResidualLayer_DAFL(filters=128, block_type=self.block_type[ver], num_units=self.num_units[ver][1], name='conv_3')
        self.conv_4 = ResidualLayer_DAFL(filters=256, block_type=self.block_type[ver], num_units=self.num_units[ver][2], name='conv_4')
        self.conv_5 = ResidualLayer_DAFL(filters=512, block_type=self.block_type[ver], num_units=self.num_units[ver][3], name='conv_5')

        self.glb_pool = keras.layers.GlobalAvgPool2D(name='glb_pool')
                
        if self.return_logits is False:
            self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax,     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits',     kernel_regularizer=self.l2_regu, bias_regularizer=self.l2_regu)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.glb_pool(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def build(self):
        super().build(input_shape=[None, *self.input_dim])
    
    def summary(self, with_graph:bool=False, **kwargs):
        inputs = keras.layers.Input(shape=self.input_dim)
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

    def get_config(self):
        return {
            'ver': self.ver,
            'input_dim': self.input_dim,
            'num_classes': self.num_classes,
            'return_logits': self.return_logits,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == '__main__':
    net = ResNet_DAFL(ver=34, input_dim=[32, 32, 3], num_classes=10, return_logits=False)
    net.build()
    net.summary(expand_nested=True, line_length=100)

    net = ResNet_DAFL(ver=50, input_dim=[32, 32, 3], num_classes=10, return_logits=False)
    net.build()
    net.summary(expand_nested=True, line_length=100)