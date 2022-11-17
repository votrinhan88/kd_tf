import warnings
from typing import List, Union, Tuple
import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.distillers.DataFreeDistiller import DataFreeDistiller
else:
    from .DataFreeDistiller import DataFreeDistiller, DataFreeGenerator

class ConditionalDataFreeGenerator(DataFreeGenerator):
    """Review GAN to CGAN to modify.
    """
    pass

    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[32, 32, 3],
                 embed_dim:int=50,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 dafl_batchnorm:bool=True):
        assert isinstance(dafl_batchnorm, bool), '`dafl_batchnorm` must be of type bool'
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'

        keras.Model.__init__()
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.dafl_batchnorm = dafl_batchnorm

        self._BASE_DIM = [self.image_dim[0]//4, self.image_dim[1]//4]
        if self.dafl_batchnorm is True:
            self._EPSILONS = [1e-5, 0.8, 0.8, 1e-5]
            self._MOMENTUM = 0.9
        elif self.dafl_batchnorm is False:
            self._EPSILONS = [keras.layers.BatchNormalization().epsilon]*4 # 1e-3
            self._MOMENTUM = keras.layers.BatchNormalization().momentum # 0.99

        # Latent branch, converted from DAFL's generator
        self.latent_branch = keras.Sequential([
            keras.layers.Dense(units=self._BASE_DIM[0] * self._BASE_DIM[1] * 128),
            keras.layers.Reshape(target_shape=(self._BASE_DIM[0], self._BASE_DIM[1], 128)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

        # Conditional label branch
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.label_branch = keras.Sequential([
            keras.layers.Dense(units=self.embed_dim),
            keras.layers.Dense(units=tf.math.reduce_prod(self.base_dim[0:-1]), use_bias=False),
            keras.layers.Reshape(target_shape=(*self.base_dim[0:-1], 1)),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

        # Main branch: concat both branches and upsample
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.concat = keras.layers.Concatenate()

        
        self.conv_block_0 = keras.Sequential(
            layers=[keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[0])],
            name='conv_block_0'
        )

        self.upsamp_1 = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='upsamp_1')
        self.conv_block_1 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[1]),
                keras.layers.LeakyReLU(alpha=0.2)
            ],
            name='conv_block_1'
        )
        
        self.upsamp_2 = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='upsamp_2')
        self.conv_block_2 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[2]),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(filters=self.image_dim[2], kernel_size=3, strides=1, padding='same'),
                keras.layers.Activation(tf.nn.tanh),
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[3], center=False, scale=False)],
            name='conv_block_2'
        )

    def call(self, inputs, training:bool=False):
        # Parse inputs
        latents, labels = inputs
        # Forward
        latents = self.latent_branch(latents, training=training)
        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        labels = self.label_branch(labels, training=training)
        x = self.concat([latents, labels])
        x = self.conv_block_0(x, training=training)
        x = self.upsamp_1(x)
        x = self.conv_block_1(x, training=training)
        x = self.upsamp_2(x)
        x = self.conv_block_2(x, training=training)
        return x
    
    def build(self):
        if self.onehot_input is True:
            keras.Model.build(self, input_shape=[[None, self.latent_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            keras.Model.build(self, input_shape=[[None, self.latent_dim], [None, 1]])

    def summary(self, with_graph:bool=False, **kwargs):
        latent_inputs = keras.layers.Input(shape=[self.latent_dim])
        if self.onehot_input is True:
            label_inputs = keras.layers.Input(shape=[self.num_classes])
        elif self.onehot_input is False:
            label_inputs = keras.layers.Input(shape=[1])
        inputs = [latent_inputs, label_inputs]
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'onehot_input': self.onehot_input,
            'dafl_batchnorm': self.dafl_batchnorm,            
        })
        return config

class CDAFL(DataFreeDistiller):
    """Actually does not need to write from scratch a new class, only needs to write a
    new method for generating synthetic images.

    Originally in train_step():
        latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
        x_synth = self.generator(latent_noise, training=True)
        --> Can we wrap this in a method, and overwrite later?

    So for CGAN, should add like this in train_step():
        latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
        label = <randomly evenly-distributed labels>
        x_synth = self.generator([latent_noise, label], training=True)

        ...
        # Teacher predicts one of num_classes
        teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)
        # But we don't have to use pseudo-label anymore
        # --> One-hot loss can be changed to KLDiv/CE loss.
        ...
        # Student also predicts one of num_classes
        student_prob = self.student(x_synth, training=True)
        ...
        # Distillation is still the same
        loss_distill = self.distill_loss_fn(teacher_prob, student_prob)
    """
    pass