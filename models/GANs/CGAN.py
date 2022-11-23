# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://phamdinhkhanh.github.io/2020/08/09/ConditionalGAN.html
# https://www.youtube.com/watch?v=MAMSh5kVoec

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.GANs.GAN import GAN
    from models.GANs.utils import RepeatTensor
else:
    from .GAN import GAN
    from .utils import RepeatTensor

from typing import List, Union
import tensorflow as tf
keras = tf.keras

# TODO:
#   - Build ConditionalGeneratorEmbed as base class, inherit to Stack version
#       - Update get_config()
#   - Same thing for the discriminators
#   - Is there any methods unfinished in CGAN? Update them too.
class ConditionalGeneratorEmbed(keras.Model):
    """Conditional generator for cGAN. Conditional inputs is fed through an
    embedding layer and concatenated with shallow feature maps.
    
    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
        `base_dim`: Dimension of the shallowest feature maps. After each
            convolutional layer, each dimension is doubled the and number of filters
            is halved until `image_dim` is reached. Defaults to `[7, 7, 256]`.
        `embed_dim`: Dimension of embedding layer. Defaults to `50`.
        `num_classes`: Number of classes. Defaults to `10`.
        `onehot_input`: `onehot_input`: Flag to indicate whether the model receives
            one-hot or label encoded target classes. Defaults to `True`.
    """    
    _name = 'cGen_embed'

    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, 256],
                 embed_dim:int=50,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 **kwargs):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
            `base_dim`: Dimension of the shallowest feature maps. After each
                convolutional layer, each dimension is doubled the and number of filters
                is halved until `image_dim` is reached. Defaults to `[7, 7, 256]`.
            `embed_dim`: Dimension of embedding layer. Defaults to `50`.
            `num_classes`: Number of classes. Defaults to `10`.
            `onehot_input`: `onehot_input`: Flag to indicate whether the model receives
                one-hot or label encoded target classes. Defaults to `True`.
        """
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'

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
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input

        # Traditional latent branch
        self.latent_branch = keras.Sequential([
            keras.layers.Dense(units=tf.math.reduce_prod(self.base_dim), use_bias=False),
            keras.layers.Reshape(target_shape=self.base_dim),
            keras.layers.ReLU()
        ])

        # Conditional label branch
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        # Replace Embedding with Dense for to accept interpolated label inputs
        self.label_branch = keras.Sequential([
            keras.layers.Dense(units=self.embed_dim),
            keras.layers.ReLU(),
            keras.layers.Dense(units=tf.math.reduce_prod(self.base_dim[0:-1])),
            keras.layers.Reshape(target_shape=(*self.base_dim[0:-1], 1)),
            keras.layers.ReLU(),
        ])

        # Main branch: concat both branches and upsample
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.concat = keras.layers.Concatenate()
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
        # Parse inputs
        latents, labels = inputs
        # Forward
        latents = self.latent_branch(latents, training=training)
        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        labels = self.label_branch(labels, training=training)
        x = self.concat([latents, labels])
        for block in self.convt_block:
            x = block(x, training=training)
        return x

    def build(self):
        if self.onehot_input is True:
            super().build(input_shape=[[None, self.latent_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            super().build(input_shape=[[None, self.latent_dim], [None, 1]])

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
            'num_classes': self.num_classes
        })
        return config

class ConditionalDiscriminatorEmbed(keras.Model):
    """Conditional discriminator for cGAN. Conditional inputs is fed through an
    embedding layer and concatenated with shallow feature maps.
    
    Args:
        `image_dim`: Dimension of input image. Defaults to `[28, 28, 1]`.
        `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
            generator's. Opposite to the generator, after each convolutional layer,
            each dimension from `image_dim` is halved and the number of filters is
            doubled until `base_dim` is reached. Defaults to `[7, 7, 256]`.
        `embed_dim`: Dimension of embedding layer. Defaults to `50`.
        `num_classes`: Number of classes. Defaults to `10`.
        `return_logits`: flag to choose between return logits or probability.
            Defaults to `False`.
    """    
    _name = 'cDisc_embed'
    
    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, 256],
                 embed_dim:int=50,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=False,
                 **kwargs):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of input image. Defaults to `[28, 28, 1]`.
            `base_dim`: Dimension of the shallowest feature maps, ideally equal to the
                generator's. Opposite to the generator, after each convolutional layer,
                each dimension from `image_dim` is halved and the number of filters is
                doubled until `base_dim` is reached. Defaults to `[7, 7, 256]`.
            `embed_dim`: Dimension of embedding layer. Defaults to `50`.
            `num_classes`: Number of classes. Defaults to `10`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `False`.
        """
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in range(len(image_dim)-1)]
        for axis in range(len(dim_ratio)):
            num_conv = tf.math.log(dim_ratio[axis])/tf.math.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        super().__init__(self, name=self._name, **kwargs)
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.return_logits = return_logits

        # Conditional label branch
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        # Replace Embedding with Dense for to accept interpolated label inputs
        self.label_branch = keras.Sequential([
            keras.layers.Dense(units=self.embed_dim),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Dense(units=self.image_dim[0]*self.image_dim[1]),
            keras.layers.Reshape(target_shape=(self.image_dim[0], self.image_dim[1], 1)),
            keras.layers.LeakyReLU(alpha=0.2),
        ])

        self.concat = keras.layers.Concatenate()

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
        
        self.flatten = keras.layers.Flatten()
        if self.return_logits is False:
            self.pred = keras.layers.Dense(units=1, name='pred', activation=tf.nn.sigmoid)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, name='logits')

    def call(self, inputs, training:bool=False):
        # Parse inputs
        images, labels = inputs
        # Forward
        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        labels = self.label_branch(labels)
        x = self.concat([images, labels])
        for block in self.conv_block:
            x = block(x, training=training)
        x = self.flatten(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x
    
    def build(self):
        if self.onehot_input is True:
            super().build(input_shape=[[None, *self.image_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            super().build(input_shape=[[None, *self.image_dim], [None, 1]])

    def summary(self, with_graph:bool=False, **kwargs):
        image_inputs = keras.layers.Input(shape=self.image_dim)
        if self.onehot_input is True:
            label_inputs = keras.layers.Input(shape=[self.num_classes])
        elif self.onehot_input is False:
            label_inputs = keras.layers.Input(shape=[1])
        inputs = [image_inputs, label_inputs]
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_dim': self.image_dim,
            'base_dim': self.base_dim,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'onehot_input': self.onehot_input,
            'return_logits': self.return_logits,
        })
        return config

class ConditionalGeneratorStack(keras.Model):
    _name = 'cGen_stack'
    def __init__(self,
                 latent_dim:int=128,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, None],
                 num_classes:int=10,
                 onehot_input:bool=True,
                 **kwargs):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `128`.
            `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
            `base_dim`: _description_. Defaults to `[7, 7, None]`.
            `num_classes`: Number of classes. Defaults to `10`.
            `onehot_input`: `onehot_input`: Flag to indicate whether the model receives
                one-hot or label encoded target classes. Defaults to `True`.
        """
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'

        super().__init__(self, name=self._name, **kwargs)

        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.base_dim = base_dim
        self.embed_dim = -1              # Unused, leave as-is for compatability
        self.num_classes = num_classes
        self.onehot_input = onehot_input

        self.generator_in_channels = self.latent_dim + self.num_classes
        
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.concat = keras.layers.Concatenate()
        self.dense_0 = keras.layers.Dense(tf.math.reduce_prod(self.base_dim[0:-1])*self.generator_in_channels)
        self.lrelu_0 = keras.layers.LeakyReLU(alpha=0.2)
        self.reshape = keras.layers.Reshape((*self.base_dim[0:-1], self.generator_in_channels))
        self.conv_1  = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        self.lrelu_1 = keras.layers.LeakyReLU(alpha=0.2)
        self.conv_2  = keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        self.lrelu_2 = keras.layers.LeakyReLU(alpha=0.2)
        self.conv_3  = keras.layers.Conv2D(self.image_dim[-1], (7, 7), padding="same", activation="tanh")
    
    def call(self, inputs, training:bool=False):
        # Parse inputs
        latents, labels = inputs

        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        x = self.concat([latents, labels])

        x = self.dense_0(x)
        x = self.lrelu_0(x)
        x = self.reshape(x)
        x = self.conv_1(x)
        x = self.lrelu_1(x)
        x = self.conv_2(x)
        x = self.lrelu_2(x)
        x = self.conv_3(x)
        return x
    
    def build(self):
        if self.onehot_input is True:
            super().build(input_shape=[[None, self.latent_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            super().build(input_shape=[[None, self.latent_dim], [None, 1]])

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

class ConditionalDiscriminatorStack(keras.Model):
    _name = 'cDisc_stack'
    
    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, 256],
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=False,
                 **kwargs):
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        # Parse architecture from input dimension
        dim_ratio = [image_dim[axis]/base_dim[axis] for axis in range(len(image_dim)-1)]
        for axis in range(len(dim_ratio)):
            num_conv = tf.math.log(dim_ratio[axis])/tf.math.log(2.)
            assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
            assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
        num_conv = int(num_conv)

        super().__init__(self, name=self._name, **kwargs)
        self.image_dim = image_dim
        self.embed_dim = -1              # Unused, leave as-is for compatability
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.return_logits = return_logits

        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.repeat   = RepeatTensor(repeats=self.image_dim[0:-1])
        self.concat   = keras.layers.Concatenate()
        self.conv_0   = keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")
        self.lrelu_0  = keras.layers.LeakyReLU(alpha=0.2)
        self.conv_1   = keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")
        self.lrelu_1  = keras.layers.LeakyReLU(alpha=0.2)
        self.gmaxpool = keras.layers.GlobalMaxPooling2D()
        if self.return_logits is False:
            self.pred = keras.layers.Dense(units=1, name='pred', activation=tf.nn.sigmoid)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, name='logits')

    def call(self, inputs, training:bool=False):
        # Parse inputs
        images, labels = inputs

        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        labels = self.repeat(labels)
        x = self.concat([images, labels])
        x = self.conv_0(x)        
        x = self.lrelu_0(x)
        x = self.conv_1(x)        
        x = self.lrelu_1(x)
        x = self.gmaxpool(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def build(self):
        if self.onehot_input is True:
            super().build(input_shape=[[None, *self.image_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            super().build(input_shape=[[None, *self.image_dim], [None, 1]])

    def summary(self, with_graph:bool=False, **kwargs):
        image_inputs = keras.layers.Input(shape=self.image_dim)
        if self.onehot_input is True:
            label_inputs = keras.layers.Input(shape=[self.num_classes])
        elif self.onehot_input is False:
            label_inputs = keras.layers.Input(shape=[1])
        inputs = [image_inputs, label_inputs]
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

class CGAN(GAN):
    """Conditional Generative Adversarial Network.
    
    Args:
        `generator`: Generator model. Defaults to `Generator()`.
        `discriminator`: Discriminator model. Defaults to `Discriminator()`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            generator. Defaults to `None`.
        `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
            generator. Defaults to `None`.
        `embed_dim`: Dimension of embedding vector, leave as `None` to be parsed
            from generator. Defaults to `None`.
        `num_classes`: Number of classes, leave as `None` to be parsed from
            generator. Defaults to `None`.
    """    
    _name = 'cGAN'

    def __init__(self,
                 generator:keras.Model,
                 discriminator:keras.Model,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 num_classes:Union[None, int]=None,
                 onehot_input:Union[None, bool]=None,
                 **kwargs):
        """Initialize cGAN.
        
        Args:
            `generator`: Generator model. Defaults to `Generator()`.
            `discriminator`: Discriminator model. Defaults to `Discriminator()`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                generator. Defaults to `None`.
            `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
                generator. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from
                generator. Defaults to `None`.
        """
        super(CGAN, self).__init__(
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            image_dim=image_dim,
            **kwargs)

        if num_classes is None:
            self.num_classes:int = self.generator.num_classes
        elif num_classes is not None:
            self.num_classes = num_classes

        if onehot_input is None:
            self.onehot_input:bool = self.generator.onehot_input
        elif onehot_input is not None:
            self.onehot_input = onehot_input

    def call(self, inputs, training:bool=False):
        latents, labels = inputs
        x_synth = self.generator.call([latents, labels], training=training)
        pred = self.discriminator.call([x_synth, labels], training=training)
        return pred

    def compile(self,
                optimizer_gen:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
                optimizer_disc:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
                loss_fn:keras.losses.Loss=keras.losses.BinaryCrossentropy(),
                **kwargs):
        super(CGAN, self).compile(
            optimizer_disc = optimizer_disc,
            optimizer_gen = optimizer_gen,
            loss_fn = loss_fn,
            **kwargs)

    def train_step(self, data):
        '''
        Notation:
            label: correspoding to label in training set (0 to `num_classes - 1`)
            x: image (synthetic or real)
            y/pred: validity/prediction of image (0 for synthetic, 1 for real)
        '''
        # Unpack data
        x_real, label = data
        batch_size = tf.shape(x_real)[0]
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        # Phase 1 - Training the discriminator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        x_synth = self.generator([latent_noise, label])
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator.trainable_variables)

            pred_real = self.discriminator([x_real, label], training=True)
            pred_synth = self.discriminator([x_synth, label], training=True)

            loss_real = self.loss_fn(y_real, pred_real)
            loss_synth = self.loss_fn(y_synth, pred_synth)
        # Back-propagation
        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(loss_real, trainable_vars)        
        self.optimizer_disc.apply_gradients(zip(gradients, trainable_vars))
        gradients = tape.gradient(loss_synth, trainable_vars)        
        self.optimizer_disc.apply_gradients(zip(gradients, trainable_vars))

        # Phase 2 - Training the generator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.generator.trainable_variables)
            x_synth = self.generator([latent_noise, label], training=True)
            pred_synth = self.discriminator([x_synth, label], training=True)
            loss_gen = self.loss_fn(y_real, pred_synth)
        # Back-propagation
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_gen, trainable_vars)        
        self.optimizer_gen.apply_gradients(zip(gradients, trainable_vars))
        del tape

        # Update the metrics, configured in 'compile()'.
        self.loss_real_metric.update_state(loss_real)
        self.loss_synth_metric.update_state(loss_synth)
        self.loss_gen_metric.update_state(loss_gen)
        results = {m.name: m.result() for m in self.train_metrics}
        return results

    def test_step(self, data):
        # Unpack data
        x_real, label = data
        batch_size:int = x_real.shape[0]
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        # Test 1 - Discriminator performs on real data
        pred_real = self.discriminator([x_real, label], training=False)

        # Test 2 - Generator tries to fool discriminator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        x_synth = self.generator([latent_noise, label], training=False)
        pred_synth = self.discriminator([x_synth, label], training=False)
        
        # Update the metrics, configured in 'compile()'.
        self.accuracy_real_metric.update_state(y_true=y_real, y_pred=pred_real)
        self.accuracy_synth_metric.update_state(y_true=y_synth, y_pred=pred_synth)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

    def get_config(self):
        config = keras.Model.get_config(self)
        config.update({
            'generator_class':self.generator.__class__,
            'generator': self.generator.get_config(),
            'discriminator_class':self.discriminator.__class__,
            'discriminator': self.discriminator.get_config(),
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim,
            'num_classes': self.num_classes,
        })
        return config

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
            keras.Model.summary(self, **kwargs)

if __name__ == '__main__':
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback
    from dataloader import dataloader

    ds, info = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        batch_size_train=64,
        batch_size_test=1000,
        drop_remainder=True,
        onehot_label=True,
        with_info=True)
    class_names = info.features['label'].names

    cgen = ConditionalGeneratorEmbed(
        latent_dim=100,
        image_dim=[28, 28, 1],
        base_dim=[7, 7, 256],
        # embed_dim=50,
        num_classes=10,
        onehot_input=True
    )
    cgen.build()

    cdisc = ConditionalDiscriminatorEmbed(
        image_dim=[28, 28, 1],
        base_dim=[7, 7, 256],
        # embed_dim=50,
        num_classes=10,
        onehot_input=True
    )
    cdisc.build()
    
    cgan = CGAN(
        generator=cgen, discriminator=cdisc
    )
    cgan.build()
    cgan.summary(with_graph=True, line_length=120, expand_nested=True)
    cgan.compile()

    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}.csv',
        append=True)
    
    gif_maker = MakeConditionalSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}.gif', 
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    interpolater = MakeInterpolateSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}_itpl.gif', 
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    slerper = MakeInterpolateSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}_itpl_slerp.gif',
        itpl_method='slerp',
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    cgan.fit(
        x=ds['train'],
        epochs=50,
        callbacks=[csv_logger, gif_maker, interpolater, slerper],
        validation_data=ds['test'],
    )