# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/
# https://phamdinhkhanh.github.io/2020/08/09/ConditionalGAN.html
# https://www.youtube.com/watch?v=MAMSh5kVoec

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.GANs.traditional_gan import GenerativeAdversarialNetwork
    from models.GANs.utils import RepeatTensor
else:
    from .traditional_gan import GenerativeAdversarialNetwork
    from .utils import RepeatTensor

from typing import List, Union
import tensorflow as tf
keras = tf.keras

class ConditionalGeneratorEmbed(keras.Model):
    # TODO: add onehot_input
    """Conditional generator for cGAN. Conditional inputs is fed through an
    embedding layer and concatenated with shallow feature maps.
    
    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
        `embed_dim`: Dimension of embedding layer. Defaults to `50`.
        `num_classes`: Number of classes. Defaults to `10`.
    """    
    _name = 'cGenerator_embed'

    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[28, 28, 1],
                 embed_dim:int=50,
                 num_classes:int=10,
                 **kwargs):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
            `embed_dim`: Dimension of embedding layer. Defaults to `50`.
            `num_classes`: Number of classes. Defaults to `10`.
        """        
        super().__init__(self, name=self._name, **kwargs)

        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Traditional generator branch
        self.latent_branch = keras.Sequential(
            layers=[
                keras.layers.Dense(units=128*7*7),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Reshape(target_shape=(7, 7, 128))
            ],
            name='latent_branch'
        )

        # Additional label branch (conditional)
        # self.label_input = keras.layers.Input(shape=(1,))   # Sparse
        self.label_branch = keras.Sequential(
            layers=[
                keras.layers.Embedding(
                    input_dim=self.num_classes,
                    output_dim=self.embed_dim,
                    input_length=1), # Embed for categorical input
                keras.layers.Dense(units=7*7),
                keras.layers.Reshape(target_shape=(7, 7, 1))
            ],
            name='label_branch'
        )

        # Main branch: concat both branches and upsample twice to 28*28
        self.concat = keras.layers.Concatenate(name='concat')
        self.main_branch = keras.Sequential(
            layers=[
                keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(filters=self.image_dim[-1], kernel_size=(7, 7), padding='same'),
                keras.layers.Activation(activation=tf.nn.tanh)
            ],
            name='main_branch'
        )

    def call(self, inputs):
        # Parse inputs
        latents, labels = inputs
        # Forward
        latents = self.latent_branch(latents)
        labels = self.label_branch(labels)
        x = self.concat([latents, labels])
        x = self.main_branch(x)
        return x

    def build(self):
        super().build(input_shape=[[None, self.latent_dim], [None, 1]])
        latent_inputs = keras.layers.Input(shape=[self.latent_dim])
        label_inputs = keras.layers.Input(shape=[1])
        self.call([latent_inputs, label_inputs])

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
        `embed_dim`: Dimension of embedding layer. Defaults to `50`.
        `num_classes`: Number of classes. Defaults to `10`.
        `return_logits`: flag to choose between return logits or probability.
            Defaults to `False`.
    """    
    # TODO: add onehot_input
    _name = 'cDiscriminator_stack'
    
    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 embed_dim:int=50,
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of input image. Defaults to `[28, 28, 1]`.
            `embed_dim`: Dimension of embedding layer. Defaults to `50`.
            `num_classes`: Number of classes. Defaults to `10`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `False`.
        """
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super().__init__(self, name=self._name, **kwargs)
        self.image_dim = image_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.label_branch = keras.Sequential(
            layers=[
                keras.layers.Embedding(
                    input_dim=self.num_classes,
                    output_dim=self.embed_dim,
                    input_length=1),
                keras.layers.Dense(units=self.image_dim[0]*self.image_dim[1]),
                keras.layers.Reshape(target_shape=(self.image_dim[0], self.image_dim[1], 1))
            ],
            name='label_branch'
        )

        self.concat = keras.layers.Concatenate(name='concat')
        self.main_branch = keras.Sequential(
            layers=[
                keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Flatten(),
                keras.layers.Dropout(rate=0.4),
            ],
            name='main_branch'
        )
        if self.return_logits is False:
            self.pred = keras.layers.Dense(units=1, name='pred', activation=tf.nn.sigmoid)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, name='logits')

    def call(self, inputs, training:bool=False):
        # Parse inputs
        images, labels = inputs
        # Forward
        labels = self.label_branch(labels)
        x = self.concat([images, labels])
        x = self.main_branch(x, training=training)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x
    
    def build(self):
        super().build(input_shape=[[None, *self.image_dim], [None, 1]])
    
    def summary(self, **kwargs):
        image_inputs = keras.layers.Input(shape=self.image_dim)
        label_inputs = keras.layers.Input(shape=(1,))
        inputs = [image_inputs, label_inputs]
        model = keras.Model(inputs=inputs, outputs=self.call(inputs))
        return model.summary(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'image_dim': self.image_dim,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes
        })
        return config

class ConditionalGeneratorStack(keras.Model):
    _name = 'cGenerator_stack'
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
        self.conv_3  = keras.layers.Conv2D(1, (7, 7), padding="same", activation="tanh")
    
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
    _name = 'cDiscriminator_stack'
    
    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=False,
                 **kwargs):
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
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

class ConditionalGenerativeAdversarialNetwork(GenerativeAdversarialNetwork):
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
                 embed_dim:Union[None, int]=None,
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
            `embed_dim`: Dimension of embedding vector, leave as `None` to be parsed
                from generator. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from
                generator. Defaults to `None`.
        """
        keras.Model.__init__(self, name=self._name, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

        if latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim
        elif latent_dim is not None:
            self.latent_dim = latent_dim

        if image_dim is None:
            self.image_dim:List[int] = self.generator.image_dim
        elif image_dim is not None:
            self.image_dim = image_dim

        if embed_dim is None:
            self.embed_dim:int = self.generator.embed_dim
        elif embed_dim is not None:
            self.embed_dim = embed_dim

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
        super(ConditionalGenerativeAdversarialNetwork, self).compile(
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
        batch_size:int = x_real.shape[0]
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
            'embed_dim': self.embed_dim,
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
            super().summary(**kwargs)

class ConditionalGenerativeAdversarialNetwork_keras(ConditionalGenerativeAdversarialNetwork):
    """Runs with keras default implementation.
    """    
    def train_step(self, data):
        '''
        Notation:
            label: correspoding to label in training set (0 to `num_classes - 1`)
            x: images (synthetic or real)
            y: validity of image (0 for synthetic, 1 for real)
        '''
        # Unpack data
        x_real, label = data
        batch_size:int = x_real.shape[0]
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        image_one_hot_labels = label[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=tf.math.reduce_prod(self.image_dim[0:-1])
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, *self.image_dim[0:-1], self.num_classes)
        )
        x_real = tf.concat([x_real, image_one_hot_labels], -1)


        # Phase 1 - Training the discriminator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        random_vector_labels = tf.concat(
            [latent_noise, label], axis=1
        )

        x_synth = self.generator(random_vector_labels)
        x_synth = tf.concat([x_synth, image_one_hot_labels], -1)

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator.trainable_variables)

            pred_real = self.discriminator(x_real, training=True)
            pred_synth = self.discriminator(x_synth, training=True)

            loss_real = self.loss_fn(y_real, pred_real)
            loss_synth = self.loss_fn(y_synth, pred_synth)
        # Back-propagation
        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(loss_real, trainable_vars)        
        self.optimizer_disc.apply_gradients(zip(gradients, trainable_vars))
        gradients = tape.gradient(loss_synth, trainable_vars)        
        self.optimizer_disc.apply_gradients(zip(gradients, trainable_vars))
        del tape

        # Phase 2 - Training the generator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        random_vector_labels = tf.concat(
            [latent_noise, label], axis=1
        )

        # label_synth = tf.random.uniform(shape=[batch_size, 1], minval=0, maxval=self.num_classes-1, dtype=tf.int32)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.generator.trainable_variables)
            x_synth = self.generator(random_vector_labels, training=True)
            x_synth = tf.concat([x_synth, image_one_hot_labels], -1)
            pred_synth = self.discriminator(x_synth, training=True)
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

        image_one_hot_labels = label[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=tf.math.reduce_prod(self.image_dim[0:-1])
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, *self.image_dim[0:-1], self.num_classes)
        )
        x_real = tf.concat([x_real, image_one_hot_labels], -1)

        # Test 1 - Discriminator performs on real data
        pred_real = self.discriminator(x_real, training=False)

        # Test 2 - Generator tries to fool discriminator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        random_vector_labels = tf.concat(
            [latent_noise, label], axis=1
        )
        # label_synth = tf.random.uniform(shape=[batch_size, 1], minval=0, maxval=self.num_classes-1, dtype=tf.int32)
        x_synth = self.generator(random_vector_labels)
        x_synth = tf.concat([x_synth, image_one_hot_labels], -1)
        pred_synth = self.discriminator(x_synth, training=False)
        
        # Update the metrics, configured in 'compile()'.
        self.accuracy_real_metric.update_state(y_true=y_real, y_pred=pred_real)
        self.accuracy_synth_metric.update_state(y_true=y_synth, y_pred=pred_synth)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

if __name__ == '__main__':
    import tensorflow_datasets as tfds
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback

    def def_gen_disc_stack_keras(batch_size = 64,
                                 num_channels = 1,
                                 num_classes = 10,
                                 image_size = 28,
                                 latent_dim = 128):
        generator_in_channels = latent_dim + num_classes
        discriminator_in_channels = num_channels + num_classes

        generator = keras.Sequential(
            [
                keras.layers.InputLayer((generator_in_channels,)),
                # We want to generate 128 + num_classes coefficients to reshape into a
                # 7x7x(128 + num_classes) map.
                keras.layers.Dense(7 * 7 * generator_in_channels),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Reshape((7, 7, generator_in_channels)),
                keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(1, (7, 7), padding="same", activation="tanh"),
            ],
            name="generator",
        )

        discriminator = keras.Sequential(
            [
                keras.layers.InputLayer((28, 28, discriminator_in_channels)),
                keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.GlobalMaxPooling2D(),
                keras.layers.Dense(1, activation='sigmoid'),
            ],
            name="discriminator",
        )

        return generator, discriminator

    ds = tfds.load('mnist', as_supervised=True)
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = (x - 0.5)/0.5
        y = tf.cast(y, tf.int32)
        y = tf.one_hot(indices=y, depth=10)
        return x, y
    ds['train'] = ds['train'].take(1000).map(preprocess).shuffle(60000).batch(64, drop_remainder=True).prefetch(1)
    ds['test'] = ds['test'].map(preprocess).batch(500, drop_remainder=True).prefetch(1)

    cgen = ConditionalGeneratorStack()
    cgen.build()

    cdisc = ConditionalDiscriminatorStack()
    cdisc.build()
    
    cgan = ConditionalGenerativeAdversarialNetwork(
        generator=cgen, discriminator=cdisc, embed_dim=-1
    )
    cgan.build()
    cgan.summary(line_length=120, expand_nested=True)

    cgan.compile(
        optimizer_gen=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5),
        optimizer_disc=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5)
    )
    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}.csv',
        append=True)
    
    gif_maker = MakeConditionalSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}.gif', 
        postprocess_fn=lambda x:(x+1)/2
    )
    interpolater = MakeInterpolateSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}_itpl.gif', 
        postprocess_fn=lambda x:(x+1)/2
    )
    cgan.fit(
        x=ds['train'],
        epochs=1,
        verbose=1,
        callbacks=[csv_logger, gif_maker, interpolater],
        validation_data=ds['test'],
    )