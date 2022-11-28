from typing import Literal, List

import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.classifiers.ResNet import (
        ResidualBasicBlock, ResidualBottleneck, ResidualLayer, ResNet)
    from models.classifiers.utils import Identity
else:
    from .utils import Identity
    from .ResNet import (
        ResidualBasicBlock, ResidualBottleneck, ResidualLayer, ResNet)

# Equivalent hyperparameters for BatchNorm layer from PyTorch
EPSILON = 1e-5
MOMENTUM = 0.9

class ResidualBasicBlock_DAFL(ResidualBasicBlock):
    def __init__(self,
                 filters:int=64,
                 strides:int=1,
                 first_block:bool=False,
                 activation="relu",
                 **kwargs):
        keras.Model.__init__(self, **kwargs)
        self.filters = filters
        self.strides = strides
        self.activation = activation
        self.first_block = first_block # Unused

        self.activation_layer = keras.activations.get(activation)
        self.l2_regu = keras.regularizers.L2(l2=5e-4)

        self.main_layers = keras.Sequential([
            keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False, kernel_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
        ])

        if self.strides == 1:
            self.skip_layers = keras.Sequential([Identity()])
        elif self.strides > 1:
            self.skip_layers = keras.Sequential([
                keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=self.strides, padding='same', use_bias=False, kernel_regularizer=self.l2_regu),
                keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
            ])

        self.connect = keras.layers.Activation(self.activation)
        
class ResidualBottleneck_DAFL(ResidualBottleneck):
    def __init__(self,
                 filters:int=64,
                 strides:int=1,
                 first_block:bool=False,
                 activation="relu",
                 **kwargs):
        keras.Model.__init__(self, **kwargs)
        self.filters = filters
        self.strides = strides
        self.first_block = first_block
        self.activation = activation
        self.l2_regu = keras.regularizers.L2(l2=5e-4)

        self.main_layers = keras.Sequential([
            keras.layers.Conv2D(filters=self.filters, kernel_size=1, strides=1, padding='valid', use_bias=False, kernel_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=self.strides, padding='same', use_bias=False, kernel_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON),
            keras.layers.Activation(self.activation),
            keras.layers.Conv2D(filters=4*self.filters, kernel_size=1, strides=1, padding='valid', use_bias=False, kernel_regularizer=self.l2_regu),
            keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
        ])

        if (self.strides == 1) & (self.first_block is False):
            self.skip_layers = Identity()
        elif (self.strides > 1) | (self.first_block is True):
            self.skip_layers = keras.Sequential([
                keras.layers.Conv2D(filters=4*self.filters, kernel_size=1, strides=self.strides, padding='valid', use_bias=False, kernel_regularizer=self.l2_regu),
                keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON)
            ])

        self.connect = keras.layers.Activation(self.activation)

class ResidualLayer_DAFL(ResidualLayer):
    def __init__(self,
                 filters:int,
                 block_type:Literal['small', 'large'],
                 num_units:int,
                 first_block:bool=False,
                 activation="relu",
                 **kwargs):
        assert isinstance(first_block, bool), '`first_block` must be of type bool.'
        assert block_type in ['small', 'large'], "`block_type` must be one of 'small', 'large'."

        keras.Model.__init__(self, **kwargs)
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

class ResNet_DAFL(ResNet):
    _name = 'ResNet-DAFL'
    def __init__(self,
                 ver:Literal[18, 34, 50, 101, 152]=18,
                 input_dim:List[int]=[32, 32, 3],
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        assert ver in [18, 34, 50, 101, 152], f'`ver` must be of [18, 34, 50, 101, 152].'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        keras.Model.__init__(self, name=f'{self._name}-{ver}', **kwargs)
        self.ver = ver
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.return_logits = return_logits
        self.l2_regu = keras.regularizers.L2(l2=5e-4)

        self.conv_1 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(64, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=self.l2_regu),
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
            self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax, kernel_regularizer=self.l2_regu)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits', kernel_regularizer=self.l2_regu)

        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid, kernel_regularizer=self.l2_regu)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax, kernel_regularizer=self.l2_regu)
        elif self.return_logits is True:
            self.pred = keras.layers.Dense(units=self.num_classes, name='pred', kernel_regularizer=self.l2_regu)

if __name__ == '__main__':
    from dataloader import dataloader

    def train_ResNetDAFL_cifar10(ver:int=34):
        IMAGE_DIM = [32, 32, 3]
        NUM_CLASSES = 10
        BATCH_SIZE_TEACHER = 128
        NUM_EPOCHS_TEACHER = 200

        OPTIMIZER_TEACHER = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9) # 1e-1 to 1e-2 to 1e-3

        print(' ResNet_DAFL on CIFAR-10 with DAFL settings '.center(80,'#'))

        def augmentation_fn(x):
            x = tf.pad(tensor=x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
            x = tf.image.random_crop(value=x, size=[tf.shape(x)[0], *IMAGE_DIM])
            x = tf.image.random_flip_left_right(image=x)
            return x
        ds = dataloader(
            dataset='cifar10',
            augmentation_fn=augmentation_fn,
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
            batch_size_test=1024
        )
        
        net:keras.Model = ResNet_DAFL(ver=ver, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        net.build()

        net.compile(
            metrics=['accuracy'], 
            optimizer=OPTIMIZER_TEACHER,
            loss=keras.losses.SparseCategoricalCrossentropy())

        def schedule(epoch:int, learing_rate:float):
            if epoch in [80, 120]:
                learing_rate = learing_rate*0.1
            return learing_rate
        lr_scheduler = keras.callbacks.LearningRateScheduler(schedule)
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
            epochs=NUM_EPOCHS_TEACHER,
            callbacks=[best_callback, lr_scheduler, csv_logger],
            validation_data=ds['test']
        )
        net.load_weights(filepath=f'./logs/{net.name}_best.h5')
        net.evaluate(ds['test'])

    train_ResNetDAFL_cifar10()