from typing import List, Union
import tensorflow as tf
keras = tf.keras

class Generator(keras.Model):
    """Generator for Generative Adversarial Networks.
            
    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.

    https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
    """
    _name = 'Gen'

    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[28, 28, 1],
                 **kwargs):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[28, 28, 1]`.
        """                
        super(Generator, self).__init__(self, name=self._name, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        self.dense_0 = keras.layers.Dense(units=256, name='dense_0')
        self.lrelu_0 = keras.layers.LeakyReLU(alpha=0.2, name='lrelu_0')
        self.bnorm_0 = keras.layers.BatchNormalization(momentum=0.8, name='bnorm_0')
        self.dense_1 = keras.layers.Dense(units=512, name='dense_1')
        self.lrelu_1 = keras.layers.LeakyReLU(alpha=0.2, name='lrelu_1')
        self.bnorm_1 = keras.layers.BatchNormalization(momentum=0.8, name='bnorm_1')
        self.dense_2 = keras.layers.Dense(units=1024, name='dense_2')
        self.lrelu_2 = keras.layers.LeakyReLU(alpha=0.2, name='lrelu_2')
        self.bnorm_2 = keras.layers.BatchNormalization(momentum=0.8, name='bnorm_2')
        self.dense_3 = keras.layers.Dense(units=tf.math.reduce_prod(self.image_dim), name='dense_3')
        self.tanh    = keras.layers.Activation(activation=tf.nn.tanh, name='tanh')
        self.reshape = keras.layers.Reshape(self.image_dim, name='reshape')
    
    def call(self, inputs, training:bool=False):
        x = self.dense_0(inputs)
        x = self.lrelu_0(x)
        x = self.bnorm_0(x, training=training)
        x = self.dense_1(x)
        x = self.lrelu_1(x)
        x = self.bnorm_1(x, training=training)
        x = self.dense_2(x)
        x = self.lrelu_2(x)
        x = self.bnorm_2(x, training=training)
        x = self.dense_3(x)
        x = self.tanh(x)
        x = self.reshape(x)
        return x

    def build(self):
        super(Generator, self).build(input_shape=[None, self.latent_dim])
        inputs = keras.layers.Input(shape=self.latent_dim)
        self.call(inputs)

    def get_config(self):
        config = super(Generator, self).get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim
        })
        return config

class Discriminator(keras.Model):
    """Discriminator for Generative Adversarial Networks.

    Args:
        `image_dim`: Dimension of input image. Defaults to `[28, 28, 1]`.
        `return_logits`: flag to choose between return logits or probability.
            Defaults to `False`.

    https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
    """    
    _name = 'Disc'
    
    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 return_logits:bool=False,
                 **kwargs):
        """Initialize discriminator.
        
        Args:
            `image_dim`: Dimension of input image. Defaults to `[28, 28, 1]`.
            `return_logits`: flag to choose between return logits or probability.
                Defaults to `False`.
        """
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        super(Discriminator, self).__init__(self, name=self._name, **kwargs)
        self.image_dim=image_dim
        self.return_logits = return_logits

        self.flatten = keras.layers.Flatten(name='flatten')
        self.dense_0 = keras.layers.Dense(units=512, name='dense_0')
        self.lrelu_0 = keras.layers.LeakyReLU(alpha=0.2, name='lrelu_0')
        self.dense_1 = keras.layers.Dense(units=256, name='dense_1')
        self.lrelu_1 = keras.layers.LeakyReLU(alpha=0.2, name='lrelu_1')

        if self.return_logits is False:
            self.pred = keras.layers.Dense(units=1, name='pred', activation=tf.nn.sigmoid)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, name='logits')

    def call(self, inputs, training:bool=False):
        x = self.flatten(inputs)
        x = self.dense_0(x)
        x = self.lrelu_0(x)
        x = self.dense_1(x)
        x = self.lrelu_1(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x
    
    def build(self):
        super(Discriminator, self).build(input_shape=[None, *self.image_dim])
        inputs = keras.layers.Input(shape=self.image_dim)
        self.call(inputs)

    def get_config(self):
        config = super(Discriminator, self).get_config()
        config.update({
            'image_dim': self.image_dim,
            'return_logits': self.return_logits
        })
        return config

class GAN(keras.Model):
    """Generative Adversarial Networks.
    DOI: 10.48550/arXiv.1406.2661

    Args:
        `generator`: Generator model. Defaults to `Generator()`.
        `discriminator`: Discriminator model. Defaults to `Discriminator()`.
        `latent_dim`: Dimension of latent space, leave `None` to be parsed from
            generator. Defaults to `None`.
        `image_dim`: Dimension of synthetic image, leave `None` to be parsed from
            generator. Defaults to `None`.
    """    
    _name = 'GAN'

    def __init__(self,
                 generator:keras.Model,
                 discriminator:keras.Model,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 **kwargs):
        """Initialize GAN.
        
        Args:
            `generator`: Generator model. Defaults to `Generator()`.
            `discriminator`: Discriminator model. Defaults to `Discriminator()`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                generator. Defaults to `None`.
            `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
                generator. Defaults to `None`.
        """
        super(GAN, self).__init__(self, name=self._name, **kwargs)
        self.generator = generator
        self.discriminator = discriminator

        if latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim
        elif latent_dim is not None:
            self.latent_dim = latent_dim

        if image_dim is None:
            self.image_dim:int = self.generator.image_dim
        elif image_dim is not None:
            self.image_dim = image_dim

    def call(self, inputs, training:bool=False):
        x = self.generator(inputs, training=training)
        x = self.discriminator(x, training=training)
        return x

    def compile(self,
                optimizer_disc:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                optimizer_gen:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                loss_fn:keras.losses.Loss=keras.losses.BinaryCrossentropy(),
                **kwargs):
        """Compile GAN.
        
        Args:
            `optimizer_disc`: Optimizer for discriminator.
                Defaults to `keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)`.
            `optimizer_gen`: Optimizer for generator.
                Defaults to `keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)`.
            `loss_fn`: Loss function.
                Defaults to `keras.losses.BinaryCrossentropy()`.
        """                
        super(GAN, self).compile(**kwargs)
        self.optimizer_disc = optimizer_disc
        self.optimizer_gen = optimizer_gen
        self.loss_fn = loss_fn

        # Metrics
        self.loss_real_metric = keras.metrics.Mean(name='loss_real')
        self.loss_synth_metric = keras.metrics.Mean(name='loss_synth')
        self.loss_gen_metric = keras.metrics.Mean(name='loss_gen')
        self.accuracy_real_metric = keras.metrics.BinaryAccuracy(name='accuracy_real')
        self.accuracy_synth_metric = keras.metrics.BinaryAccuracy(name='accuracy_synth')

    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:        
        return [self.loss_real_metric, self.loss_synth_metric, self.loss_gen_metric]
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        return [self.accuracy_real_metric, self.accuracy_synth_metric]
    
    def build(self):
        super(GAN, self).build(input_shape=[None, self.latent_dim])
        inputs = keras.layers.Input(shape=[self.latent_dim])
        self.call(inputs)

    def train_step(self, data):
        # Unpack data
        x_real, _ = data
        batch_size = tf.shape(x_real)[0]
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        # Phase 1 - Training the discriminator
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator.trainable_variables)
    
            latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
            x_synth = self.generator(latent_noise, training=True)

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
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.generator.trainable_variables)

            latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
            x_synth = self.generator(latent_noise, training=True)
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
        x_real, _ = data
        batch_size:int = x_real.shape[0]
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        # Test 1 - Discriminator's performance on real images
        pred_real = self.discriminator(x_real)
        
        # Test 2 - Discriminator's performance on synthetic images
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        pred_synth = self.discriminator(self.generator(latent_noise))

        # Update the metrics, configured in 'compile()'.
        self.accuracy_real_metric.update_state(y_true=y_real, y_pred=pred_real)
        self.accuracy_synth_metric.update_state(y_true=y_synth, y_pred=pred_synth)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

    def get_config(self):
        config = super(GAN, self).get_config()
        config.update({
            'generator_class':self.generator.__class__,
            'generator': self.generator.get_config(),
            'discriminator_class':self.discriminator.__class__,
            'discriminator': self.discriminator.get_config(),
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim
        })
        return config

    @classmethod
    def from_config(cls, config:dict, custom_objects=None):
        config.update({
            'generator':config['generator_class'].from_config(config['generator']),
            'discriminator':config['discriminator_class'].from_config(config['discriminator'])
        })
        for key in ['generator_class', 'discriminator_class']:
            config.pop(key, None)
        return super(GAN, cls).from_config(config, custom_objects)

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.GANs.utils import MakeSyntheticGIFCallback
    from dataloader import dataloader

    ds = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        batch_size_train=128,
        batch_size_test=1000,
        drop_remainder=True)

    gen = Generator()
    gen.build()

    disc = Discriminator()
    disc.build()

    gan = GAN(generator=gen, discriminator=disc)
    gan.build()
    gan.summary(expand_nested=True)
    gan.compile()

    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{gan.name}_{gan.generator.name}_{gan.discriminator.name}.csv',
        append=True
    )
    gif_maker = MakeSyntheticGIFCallback(
        nrows=5, ncols=5,
        postprocess_fn=lambda x:(x+1)/2
    )
    gan.evaluate(ds['test'])
    gan.fit(
        x=ds['train'],
        epochs=50,
        verbose=1,
        callbacks=[csv_logger, gif_maker],
        validation_data=ds['test'],
    )
