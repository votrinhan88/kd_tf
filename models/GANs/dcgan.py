if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[0])))))
    from models.GANs.traditional_gan import GenerativeAdversarialNetwork
else:
    from .traditional_gan import GenerativeAdversarialNetwork

from typing import List
import tensorflow as tf
keras = tf.keras

def get_sequential_dcgan(latent_dim:int=128) -> keras.Sequential:
    """DCGAN implementation from Keras.
    https://keras.io/examples/generative/dcgan_overriding_train_step/

    Args:
        latent_dim (int, optional): Dimension of latent space. Defaults to 128.

    Returns:
        keras.Sequential: DCGAN model
    """
    discriminator = keras.Sequential(
        layers = [
            keras.Input(shape=(64, 64, 3)),
            keras.layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator")
    generator = keras.Sequential(
        layers=[
            keras.Input(shape=(latent_dim,)),
            keras.layers.Dense(8 * 8 * 128),
            keras.layers.Reshape((8, 8, 128)),
            keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            keras.layers.LeakyReLU(alpha=0.2),
            keras.layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
        ],
        name="generator")
    dcgan = keras.Sequential(
        layers=[keras.Input(shape=(latent_dim,)),generator, discriminator],
        name='DCGAN')
    return dcgan

class DC_Discriminator(keras.Model):
    """Discriminator for DCGAN.
    """    
    _name = 'DC_Discriminator'

    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        """Initialize DC Discriminator.

        Args:
            image_dim (List[int], optional): Dimension of image. Defaults to [28, 28, 1].
        """                 
        super().__init__(self, name=self._name, *args, **kwargs)
        self.image_dim = image_dim

        self.conv2d_1     = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same', name='conv2d_1')
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.2, name='leaky_relu_1')
        self.conv2d_2     = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same', name='conv2d_2')
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.2, name='leaky_relu_2')
        self.flatten_2      = keras.layers.Flatten(name='flatten_2')
        self.dropout      = keras.layers.Dropout(rate=0.2, name='dropout')
        self.dense        = keras.layers.Dense(units=1, name='dense')

    def call(self, inputs):
        x = self.conv2d_1(inputs)
        x = self.leaky_relu_1(x)
        x = self.conv2d_2(x)
        x = self.leaky_relu_2(x)
        x = self.flatten_2(x)
        x = self.dropout(x)
        x = self.dense(x)
        return x

class DC_Generator(keras.Model):
    """Generator for DCGAN.
    """    
    def __init__(self,
                 latent_dim:List[int]=[64],
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        """Initialize DC Generator.

        Args:
            latent_dim (List[int], optional): Dimension of latent space. Defaults to [64].
            image_dim (List[int], optional): Dimension of image. Defaults to [28, 28, 1].
        """                 
        super().__init__(self, name='DC_generator', *args, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        self.dense        = keras.layers.Dense(7 * 7 * 128)
        self.reshape      = keras.layers.Reshape((7, 7, 128))
        self.conv2dT_1    = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", name='conv2dT_1')
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.2, name='leaky_relu_1')
        self.conv2dT_2    = keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", name='conv2dT_2')
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.2, name='leaky_relu_2')
        self.conv2d       = keras.layers.Conv2D(1, kernel_size=5, padding="same", name='conv2d')
        self.tanh         = keras.layers.Activation(tf.nn.tanh, name='tanh')

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv2dT_1(x)
        x = self.leaky_relu_1(x)
        x = self.conv2dT_2(x)
        x = self.leaky_relu_2(x)
        x = self.conv2d(x)
        x = self.tanh(x)
        x = (x + 1)/2
        return x

class DC_GenerativeAdversarialNetwork(GenerativeAdversarialNetwork):
    """Unsupervised Representation Learning with Deep Convolutional Generative
    Adversarial Networks
    DOI: 10.48550/arXiv.1511.06434
    """    
    _name = 'DCGAN'
    discrimator_class = DC_Discriminator
    generator_class = DC_Generator

    def __init__(self,
                 generator:keras.Model=None,
                 discriminator:keras.Model=None,
                 latent_dim:List[int]=[128],
                 image_dim:List[int]=[64, 64, 3],
                 *args, **kwargs):
        """Initialize DCGAN.

        Args:
            generator (keras.Model, optional): Generator model. Defaults to None.
            discriminator (keras.Model, optional): Discriminator model. Defaults to None.
            latent_dim (List[int], optional): Dimension of latent space. Defaults to [128].
            image_dim (List[int], optional): Dimension of image. Defaults to [64, 64, 3].
        """                 
        super().__init__(generator=generator,
                         discriminator=discriminator,
                         latent_dim=latent_dim,
                         image_dim=image_dim,
                         *args, **kwargs)

if __name__ == '__main__':
    from models.GANs.utils import PlotSyntheticCallback

    # Hyperparameters
    ## Models
    IMAGE_DIM = [28, 28, 1]
    LATENT_DIM = [128]
    ## Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    BETA_1 = 0.5 # For Adam Optimizer
    ## Plotting
    NUM_SYNTHETIC = 10
    
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = tf.expand_dims(x_train.astype('float32')/255, axis = -1)
    x_test = tf.expand_dims(x_test.astype('float32')/255, axis = -1)

    dcgan = get_sequential_dcgan()
    dcgan.summary(expand_nested=True)

    dcgan = DC_GenerativeAdversarialNetwork(
        latent_dim=LATENT_DIM,
        image_dim=IMAGE_DIM)
    dcgan.build()
    dcgan.summary()
    dcgan.compile(
        disc_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1),
        gen_optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True))
    dcgan.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=[PlotSyntheticCallback(num_epochs=NUM_EPOCHS, num_synthetic=NUM_SYNTHETIC)])