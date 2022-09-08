from typing import List
import tensorflow as tf
keras = tf.keras

class Discriminator(keras.Model):
    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        super().__init__(self, name='discriminator', *args, **kwargs)
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

class Generator(keras.Model):
    def __init__(self,
                 latent_dim:List[int]=[64],
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        super().__init__(self, name='generator', *args, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        self.dense_1      = keras.layers.Dense(units=128, name='dense_1')
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_1')
        self.dense_2      = keras.layers.Dense(units=256, name='dense_2')
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.1, name='leaky_relu_2')
        self.dense_3      = keras.layers.Dense(units=tf.math.reduce_prod(self.image_dim), name='dense_3')
        self.sigmoid      = keras.layers.Activation(tf.nn.sigmoid, name='sigmoid')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        x = self.dense_3(x)
        x = self.sigmoid(x)
        return x

class GenerativeAdversarialNetwork(keras.Model):
    def __init__(self,
                 generator:keras.Model=None,
                 discriminator:keras.Model=None,
                 latent_dim:int=64,
                 image_dim:List[int]=[28, 28, 1],
                 *args, **kwargs):
        super().__init__(self, name='Traditional GAN', *args, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim

        if generator is not None:
            self.generator = generator
        else:
            self.generator = Generator(latent_dim=self.latent_dim, image_dim=self.image_dim)

        if discriminator is not None:
            self.discriminator = discriminator
        else:
            self.discriminator = Discriminator(image_dim=self.image_dim)

    def call(self, inputs):
        x = self.generator(inputs)
        x = self.discriminator(x)
        return x

    def build(self):
        inputs = keras.layers.Input(shape=self.latent_dim)
        x = inputs
        for layer in self.layers:
            x = layer.call(x)

        super().build(input_shape=[None]+self.latent_dim)
        self.call(inputs)

    def compile(self,
                metrics,
                disc_loss:keras.losses.Loss=keras.losses.BinaryCrossentropy(from_logits=True),
                disc_optimizer:keras.optimizers.Optimizer=keras.optimizers.RMSprop(),
                GAN_loss:keras.losses.Loss=keras.losses.BinaryCrossentropy(from_logits=True),
                gen_optimizer:keras.optimizers.Optimizer=keras.optimizers.RMSprop()):
        super().compile(metrics=metrics, loss=GAN_loss)
        self.generator.compile(optimizer=gen_optimizer)
        self.discriminator.compile(loss=disc_loss, optimizer=disc_optimizer)

    def train_step(self, data):
        # Unpack data
        x_real, _ = data
        batch_size:int = x_real.shape[0]

        # Phase 1 - Training the discriminator
        '''noise = tf.random.normal(shape=[batch_size, self.latent_dim[0]])
        x_synthetic = self.generator(noise)
        x_mixed = tf.concat([x_synthetic, x_real], axis=0)
        y_mixed = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)

        self.discriminator.trainable = True
        self.discriminator.train_on_batch(x_mixed, y_mixed)'''
        noise = tf.random.normal(shape=[batch_size, self.latent_dim[0]])
        x_synthetic = self.generator(noise)
        x_mixed = tf.concat([x_synthetic, x_real], axis=0)
        y_mixed = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
        with tf.GradientTape() as tape:
            pred = self.discriminator(x_mixed, training=True)
            disc_loss = self.discriminator.loss(y_mixed, pred)           
        # Compute gradients
        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(disc_loss, trainable_vars)        
        # Update weights
        self.discriminator.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y_mixed, pred)

        # Phase 2 - Training the generator
        '''noise = tf.random.normal(shape=[batch_size, self.latent_dim[0]])
        y2_synthetic = tf.constant([[1.]] * batch_size)
        self.discriminator.trainable = False
        self.train_on_batch(noise, y2_synthetic)'''
        noise = tf.random.normal(shape=[batch_size, self.latent_dim[0]])
        y2_synthetic = tf.constant([[1.]] * batch_size)
        # self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            x2_synthetic = self.generator(noise)
            pred2 = self.discriminator(x2_synthetic, training=False)
            gen_loss = self.loss(y2_synthetic, pred2)
        # Compute gradients
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(gen_loss, trainable_vars)        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y2_synthetic, pred2)


        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "disc_loss": disc_loss,
            "gen_loss": gen_loss
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
    LEARNING_RATE = 5e-3
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
        metrics=['accuracy'],
        disc_optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
        gen_optimizer=keras.optimizers.RMSprop(learning_rate=LEARNING_RATE))
    gan.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=[PlotSyntheticCallback(num_epochs=NUM_EPOCHS, num_synthetic=NUM_SYNTHETIC)])