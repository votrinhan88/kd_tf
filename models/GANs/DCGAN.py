from typing import List
import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.GANs.GAN import Generator, Discriminator, GAN
else:
    from .GAN import Generator, Discriminator, GAN

class DC_Generator(Generator):
    """Generator for DCGAN.
    
    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
        `base_dim`: Dimension of the shallowest feature maps. After each
            convolutional layer, each dimension is doubled the and number of filters
            is halved until `image_dim` is reached. Defaults to `[7, 7, 256]`.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generative-adversarial-networks
    """
    _name = 'DCGen'

    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, 256],
                 **kwargs):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
            `base_dim`: Dimension of the shallowest feature maps. After each
                convolutional layer, each dimension is doubled the and number of filters
                is halved until `image_dim` is reached. Defaults to `[7, 7, 256]`.
        """
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in range(len(image_dim)-1)]
        for axis in range(len(dim_ratio)):
            num_conv = tf.math.log(dim_ratio[axis])/tf.math.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        keras.Model.__init__(self, name=self._name, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.base_dim = base_dim

        self.dense_0 = keras.layers.Dense(units=tf.math.reduce_prod(self.base_dim), use_bias=False, name='dense_0')
        self.reshape = keras.layers.Reshape(target_shape=self.base_dim, name='reshape')
        self.bnorm_0 = keras.layers.BatchNormalization(name='bnorm_0')
        self.relu_0  = keras.layers.ReLU(name='relu_0')

        self.convt_block = [None for i in range(num_conv)]
        for i in range(num_conv):
            block_idx = i + 1
            filters = self.base_dim[-1] // 2**(i+1)
            if i < num_conv - 1:
                self.convt_block[i] = keras.Sequential(
                    layers=[
                        keras.layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, name=f'convt_{block_idx}'),
                        keras.layers.BatchNormalization(name=f'bnorm_{block_idx}'),
                        keras.layers.ReLU(name=f'relu_{block_idx}')
                    ],
                    name=f'convt_block_{block_idx}'
                )
            elif i == num_conv - 1:
                # Last Conv2DTranspose: not use BatchNorm, replace relu with tanh
                self.convt_block[i] = keras.Sequential(
                    layers=[
                        keras.layers.Conv2DTranspose(filters=self.image_dim[-1], kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, name=f'convt_{block_idx}'),
                        keras.layers.Activation(activation=tf.nn.tanh, name=f'tanh_{block_idx}')
                    ],
                    name=f'convt_block_{block_idx}'
                )

    def call(self, inputs, training:bool=False):
        x = self.dense_0(inputs)
        x = self.reshape(x)
        x = self.bnorm_0(x, training=training)
        x = self.relu_0(x)
        for block in self.convt_block:
            x = block(x, training=training)
        return x

    def get_config(self):
        config = keras.Model.get_config(self)
        config.update({
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim,
            'base_dim': self.base_dim,
        })
        return config

class DC_Discriminator(Discriminator):
    """Discriminator for DCGAN. Ideally should have a symmetric architecture with the
    generator's.

    Args:
        `image_dim`: Dimension of image. Defaults to `[28, 28, 1]`.
        `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
            generator's. Opposite to the generator, after each convolutional layer,
            each dimension from `image_dim` is halved and the number of filters is
            doubled until `base_dim` is reached. Defaults to `[7, 7, 256]`.
        `return_logits`: flag to choose between return logits or probability.
            Defaults to `False`.
    
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html#generative-adversarial-networks
    """    
    _name = 'DCDisc'

    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, 256],
                 return_logits:bool=False,
                 **kwargs):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of image. Defaults to `[28, 28, 1]`.
            `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
                generator's. Opposite to the generator, after each convolutional layer,
                each dimension from `image_dim` is halved and the number of filters is
                doubled until `base_dim` is reached. Defaults to `[7, 7, 256]`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `False`.
        """
        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in range(len(image_dim)-1)]
        for axis in range(len(dim_ratio)):
            num_conv = tf.math.log(dim_ratio[axis])/tf.math.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        keras.Model.__init__(self, name=self._name, **kwargs)
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.return_logits = return_logits

        self.conv_block = [None for i in range(num_conv)]
        for i in range(num_conv):
            block_idx = i
            filters = self.base_dim[-1] // 2**(num_conv-1-i)
            if i == 0:
                # First Conv2D: not use BatchNorm 
                self.conv_block[i] = keras.Sequential(
                    layers=[
                        keras.layers.Conv2D(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, name=f'conv_{block_idx}'),
                        keras.layers.LeakyReLU(alpha=0.2, name=f'lrelu_{block_idx}')
                    ],
                    name=f'conv_block_{block_idx}'
                )
            elif i > 0:
                self.conv_block[i] = keras.Sequential(
                    layers=[
                        keras.layers.Conv2D(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, name=f'conv_{block_idx}'),
                        keras.layers.BatchNormalization(name=f'bnorm_{block_idx}'),
                        keras.layers.LeakyReLU(alpha=0.2, name=f'lrelu_{block_idx}')
                    ],
                    name=f'conv_block_{block_idx}'
                )

        self.flatten = keras.layers.Flatten(name='flatten')

        if self.return_logits is False:
            self.pred = keras.layers.Dense(units=1, use_bias=False, activation=tf.nn.sigmoid, name='pred')
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, use_bias=False, name='logits')

    def call(self, inputs, training:bool=False):
        x = inputs
        for block in self.conv_block:
            x = block(x, training=training)
        x = self.flatten(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def get_config(self):
        config = keras.Model.get_config(self)
        config.update({
            'image_dim': self.image_dim,
            'base_dim': self.base_dim,
            'return_logits': self.return_logits
        })
        return config

class DCGAN(GAN):
    """Unsupervised Representation Learning with Deep Convolutional Generative
    Adversarial Networks
    DOI: 10.48550/arXiv.1511.06434
    """    
    _name = 'DCGAN'
    
    def compile(self,
                optimizer_disc:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
                optimizer_gen:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
                loss_fn:keras.losses.Loss=keras.losses.BinaryCrossentropy(),
                **kwargs):
        """Compile DCGAN.
        
        Args:
            `optimizer_disc`: Optimizer for discriminator.
                Defaults to `keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)`.
            `optimizer_gen`: Optimizer for generator.
                Defaults to `keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)`.
            `loss_fn`: Loss function.
                Defaults to `keras.losses.BinaryCrossentropy()`.
        """                
        super(DCGAN, self).compile(
            optimizer_disc = optimizer_disc,
            optimizer_gen = optimizer_gen,
            loss_fn = loss_fn,
            **kwargs)

if __name__ == '__main__':
    from models.GANs.utils import MakeSyntheticGIFCallback
    from dataloader import dataloader

    ds = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        batch_size_train=128,
        batch_size_test=1000,
        drop_remainder=True)

    gen = DC_Generator(image_dim=[28, 28, 1], base_dim=[7, 7, 256])
    gen.build()

    disc = DC_Discriminator(image_dim=[28, 28, 1], base_dim=[7, 7, 256])
    disc.build()

    gan = DCGAN(generator=gen, discriminator=disc)
    gan.build()
    gan.summary(expand_nested=True)
    gan.compile()

    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.csv',
        append=True
    )
    gif_maker = MakeSyntheticGIFCallback(
        f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.gif',
        nrows=5, ncols=5,
        postprocess_fn=lambda x:(x+1)/2
    )
    gan.evaluate(ds['test'])
    gan.fit(
        x=ds['train'],
        epochs=50,
        callbacks=[csv_logger, gif_maker],
        validation_data=ds['test'],
    )
