from typing import List
import tensorflow as tf
keras = tf.keras

class Discriminator(keras.Model):
    """Discriminator for Generative Adversarial Networks
    """    
    _name = 'Discriminator'
    
    def __init__(self, image_dim:List[int]=[28, 28, 1], *args, **kwargs):
        """Initialize discriminator.

        Args:
            image_dim (List[int], optional): Dimension of image. Defaults to [28, 28, 1].
        """        
        super().__init__(self, name=self._name, *args, **kwargs)
        self.image_dim = image_dim

        self.flatten      = keras.layers.Flatten(input_shape=[None, int(tf.math.reduce_prod(image_dim))], name='flatten')
        self.dense_1      = keras.layers.Dense(units=256, name='dense_1')
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_1')
        self.dense_2      = keras.layers.Dense(units=128, name='dense_2')
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_2')
        self.dense_3      = keras.layers.Dense(units=64, name='dense_3')
        self.leaky_relu_3 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_3')
        self.dense_4      = keras.layers.Dense(units=1, name='dense_4')
        # self.sigmoid      = keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')
     
    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense_1(x)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dense_3(x)
        x = self.leaky_relu_3(x)
        x = self.dense_4(x)
        # x = self.sigmoid(x)
        return x

    def get_config(self):
        return {
            'image_dim': self.image_dim,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Generator(keras.Model):
    """Generator for Generative Adversarial Networks
    """    
    _name = 'Generator'

    def __init__(self,
                 latent_dim:List[int]=[64],
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        """Initialize generator.

        Args:
            latent_dim (List[int], optional): Dimension of latent space. Defaults to [64].
            image_dim (List[int], optional): Dimension of image. Defaults to [28, 28, 1].
        """                 
        super().__init__(self, name=self._name, *args, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        self.dense_1      = keras.layers.Dense(units=128, name='dense_1')
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_1')
        self.dense_2      = keras.layers.Dense(units=256, name='dense_2')
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_2')
        self.dense_3      = keras.layers.Dense(units=tf.math.reduce_prod(self.image_dim), name='dense_3')
        self.tanh         = keras.layers.Activation(tf.nn.tanh, name='tanh')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dense_3(x)
        x = self.tanh(x)
        x = (x + 1)/2
        return x

    def get_config(self):
        return {
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GenerativeAdversarialNetwork(keras.Model):
    """Generative Adversarial Networks
    DOI: 10.48550/arXiv.1406.2661

    Label: synthetic = 0, real = 1
    """    
    _name = 'GAN'
    discrimator_class = Discriminator
    generator_class = Generator

    def __init__(self,
                 generator:keras.Model=None,
                 discriminator:keras.Model=None,
                 latent_dim:List[int]=[64],
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        """Initialize GAN.

        Args:
            generator (keras.Model, optional): Generator model. Defaults to None.
            discriminator (keras.Model, optional): Discriminator model. Defaults to None.
            latent_dim (List[int], optional): Dimension of latent space. Defaults to [64].
            image_dim (List[int], optional): Dimension of image. Defaults to [28, 28, 1].
        """        
        super(GenerativeAdversarialNetwork, self).__init__(self, name=self._name, *args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        if self.generator is not None:
            self.generator = generator
        else:
            self.generator = self.generator_class(latent_dim=self.latent_dim, image_dim=self.image_dim)

        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = self.discrimator_class(image_dim=self.image_dim)

    def call(self, inputs):
        x = self.generator(inputs)
        x = self.discriminator(x)
        return x

    def build(self):
        super().build(input_shape=[None]+self.latent_dim)
        inputs = keras.layers.Input(shape=self.latent_dim)
        x = inputs
        for layer in self.layers:
            x = layer.call(x)

        self.call(inputs)

    def compile(self,
                disc_optimizer:keras.optimizers.Optimizer=keras.optimizers.RMSprop(),
                gen_optimizer:keras.optimizers.Optimizer=keras.optimizers.RMSprop(),
                loss_fn:keras.losses.Loss=keras.losses.BinaryCrossentropy(from_logits=True)):
        """Compile GAN.

        Args:
            disc_optimizer (keras.optimizers.Optimizer, optional): Optimizer for discriminator.
                Defaults to keras.optimizers.RMSprop().
            gen_optimizer (keras.optimizers.Optimizer, optional): Optimizer for generator.
                Defaults to keras.optimizers.RMSprop().
            loss_fn (keras.losses.Loss, optional): Loss function.
                Defaults to keras.losses.BinaryCrossentropy(from_logits=True).
        """        
        super().compile(optimizer=gen_optimizer, loss=loss_fn)
        self.discriminator.compile(optimizer=disc_optimizer, loss=loss_fn)

        self.loss_disc_metric = keras.metrics.Mean(name="loss_disc")
        self.loss_gen_metric = keras.metrics.Mean(name="loss_gen")

    @property
    def metrics(self):
        return [self.loss_disc_metric, self.loss_gen_metric]

    def train_step(self, data):
        # Unpack data
        x_real, _ = data
        batch_size:int = x_real.shape[0]

        # Phase 1 - Training the discriminator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim[0]])
        x_synthetic = self.generator(latent_noise)
        x_combined = tf.concat([x_synthetic, x_real], axis=0)
        y_combined = tf.concat([tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0)
        # Add random noise to the labels - important trick!
        # y_combined += 0.05 * tf.random.uniform(tf.shape(y_combined))

        with tf.GradientTape() as tape:
            prediction = self.discriminator(x_combined, training=True)
            # loss_disc = self.loss_fn(y_combined, prediction)           
            loss_disc = self.discriminator.loss(y_combined, prediction)           
        # Back-propagation
        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(loss_disc, trainable_vars)        
        # self.disc_optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.discriminator.optimizer.apply_gradients(zip(gradients, trainable_vars))
        

        # Phase 2 - Training the generator
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim[0]])
        y_synthetic = tf.ones((batch_size, 1))
        # self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            x_synthetic = self.generator(latent_noise)
            prediction = self.discriminator(x_synthetic, training=False)
            # loss_gen = self.loss_fn(y_synthetic, prediction)
            loss_gen = self.loss(y_synthetic, prediction)
        # Back-propagation
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_gen, trainable_vars)        
        # self.gen_optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        

        # Update the metrics, configured in 'compile()'.
        results = {m.name: m.result() for m in self.metrics}
        self.loss_disc_metric.update_state(loss_disc)
        self.loss_gen_metric.update_state(loss_gen)
        results.update({
            "loss_disc": loss_disc,
            "loss_gen": loss_gen
        })
        return results

    def summary(self):
        super().summary(expand_nested=True)

if __name__ == '__main__':
    # Change path
    import os, sys
    sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[0])))))

    from models.GANs.utils import PlotSyntheticCallback

    # Hyperparameters
    ## Models
    IMAGE_DIM = [28, 28, 1]
    LATENT_DIM = [64]
    ## Training
    NUM_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 0.002 # Optimal 0.005
    ## Plotting
    NUM_SYNTHETIC = 10

    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32').reshape(-1, 784)/255
    x_test = x_test.astype('float32').reshape(-1, 784)/255

    gan = GenerativeAdversarialNetwork(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)
    gan.build()
    gan.summary()
    gan.compile(
        disc_optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        gen_optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
    gan.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=[PlotSyntheticCallback(num_epochs=NUM_EPOCHS, num_synthetic=NUM_SYNTHETIC)])