from typing import List, Union
import tensorflow as tf
keras = tf.keras


if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.GANs.GAN import GAN
else:
    from .GAN import GAN

REAL = -1
FAKE = 1
LEARNING_RATE = 5e-5
CLIP_VALUE = 0.01
BATCH_SIZE_TRAIN = 64
N_CRITIC = 5


class WassersteinLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_mean(y_true*y_pred, axis=-1)

class ClipConstraint(keras.constraints.Constraint):
    def __init__(self, clip_value:float):
        self.clip_value = clip_value
 
    def __call__(self, weights):
        return tf.clip_by_value(
            t=weights,
            clip_value_min=-self.clip_value, 
            clip_value_max=self.clip_value)
    
    def get_config(self):
        config = super(ClipConstraint, self).get_config()
        config.update({
            'clip_value':self.clip_value,
        })
        return config

def define_generator(
        latent_dim:int=100,
        image_dim:List[int]=[28, 28, 1],
        base_dim:List[int]=[7, 7, 256],
        **kwargs):
	# init = RandomNormal(stddev=0.02)
    generator = keras.Sequential([
        # Base (25% size)
        keras.layers.Dense(units=tf.reduce_prod(base_dim)),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Reshape(base_dim),
        # Upsample (50% size)
        keras.layers.Conv2DTranspose(filters=base_dim[-1]//2, kernel_size=(4, 4), strides=(2,2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        # Upsample (100% size)
        keras.layers.Conv2DTranspose(filters=base_dim[-1]//4, kernel_size=(4, 4), strides=(2,2), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        # Out
        keras.layers.Conv2D(filters=1, kernel_size=(7, 7), activation='tanh', padding='same')
    ], **kwargs)
    generator.build(input_shape=[None, latent_dim])
    return generator

def define_critic(
        image_dim:List[int]=[28, 28, 1],
        base_dim:List[int]=[7, 7, 256],
        clip_value:float=0.01,
        **kwargs):
    weight_clipper = ClipConstraint(clip_value=clip_value)
    critic = keras.Sequential([
        # Downsample (50% size)
        keras.layers.Conv2D(filters=base_dim[-1]//2, kernel_size=(4,4), strides=(2,2), padding='same', kernel_constraint=weight_clipper),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        # Downsample (25% size)
        keras.layers.Conv2D(filters=base_dim[-1], kernel_size=(4,4), strides=(2,2), padding='same', kernel_constraint=weight_clipper),
        keras.layers.BatchNormalization(),
        keras.layers.LeakyReLU(alpha=0.2),
        # Flatten to Dense
        keras.layers.Flatten(),
        keras.layers.Dense(units=1)
    ], **kwargs)
    critic.build(input_shape=[None, *image_dim])
    return critic

class WGAN(GAN):
    _name = 'WGAN'
    def __init__(self,
                 generator:keras.Model,
                 critic:keras.Model,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 **kwargs):
        super().__init__(
            generator=generator,
            discriminator=critic,
            latent_dim=latent_dim,
            image_dim=image_dim,
            **kwargs)
        # Sync name for critic
        self.critic = self.discriminator

    def compile(self,
                optimizer_crit:keras.optimizers.Optimizer=keras.optimizers.RMSprop(learning_rate=5e-5),
                optimizer_gen:keras.optimizers.Optimizer=keras.optimizers.RMSprop(learning_rate=5e-5),
                loss_fn:keras.losses.Loss=WassersteinLoss(),
                n_critic:int=5,
                **kwargs):
        """Compile WGAN.
        
        Args:
            `optimizer_crit`: Optimizer for critic.
                Defaults to `keras.optimizers.RMSprop(learning_rate=5e-5)`.
            `optimizer_gen`: Optimizer for generator.
                Defaults to `keras.optimizers.RMSprop(learning_rate=5e-5)`.
            `loss_fn`: Loss function.
                Defaults to `WassersteinLoss()`.
            `n_critic`: ber of iterations of the critic per generator iteration.
                Defaults to `5`.
        """
        super(WGAN, self).compile(
            optimizer_disc=optimizer_crit,
            optimizer_gen=optimizer_gen,
            loss_fn=loss_fn
        )
        # Sync name for critic
        self.optimizer_crit = self.optimizer_disc

        self.n_critic = n_critic
        # Step counter
        self.step = tf.Variable(0)
    
    def synthesize_images(self, batch_size:int, training:bool=False):
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        x_synth = self.generator(latent_noise, training=training)
        return x_synth

    def train_critic(self, x_real, batch_size:int):
        y_synth = tf.ones(shape=(batch_size, 1))
        y_real = -tf.ones(shape=(batch_size, 1))

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.critic.trainable_variables)
            x_synth = self.synthesize_images(batch_size=batch_size, training=False)

            pred_real = self.critic(x_real, training=True)
            pred_synth = self.critic(x_synth, training=True)

            loss_real = self.loss_fn(y_real, pred_real)
            loss_synth = self.loss_fn(y_synth, pred_synth)
        # Back-propagation
        trainable_vars = self.critic.trainable_variables
        gradients = tape.gradient(loss_real, trainable_vars)        
        self.optimizer_crit.apply_gradients(zip(gradients, trainable_vars))
        gradients = tape.gradient(loss_synth, trainable_vars)        
        self.optimizer_crit.apply_gradients(zip(gradients, trainable_vars))
        del tape

        self.loss_real_metric.update_state(loss_real)
        self.loss_synth_metric.update_state(loss_synth)

    def train_generator(self, batch_size):
        y_real = -tf.ones(shape=(batch_size, 1))

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            x_synth = tape.watch(self.generator.trainable_variables)

            x_synth = self.synthesize_images(batch_size=batch_size, training=True)
            pred_synth = self.critic(x_synth, training=True)
            loss_gen = self.loss_fn(y_real, pred_synth)
        # Back-propagation
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_gen, trainable_vars)        
        self.optimizer_gen.apply_gradients(zip(gradients, trainable_vars))
        del tape
        
        self.loss_gen_metric.update_state(loss_gen)
    
    def train_step(self, data):
        self.step.assign_add(1)
        # Unpack data
        x_real, _ = data
        batch_size = tf.shape(x_real)[0]
        
        self.train_critic(x_real, batch_size)
        # Skip if haven't waited enough `n_critic` iterations
        tf.cond(
            pred=tf.equal(self.step % self.n_critic, 0),
            true_fn=lambda:self.train_generator(batch_size),
            false_fn=lambda:(None)
        )

        results = {m.name: m.result() for m in self.train_metrics}
        return results

    def test_step(self, data):
        # Unpack data
        x_real, _ = data
        batch_size:int = x_real.shape[0]
        y_synth = tf.ones(shape=(batch_size, 1))
        y_real = -tf.ones(shape=(batch_size, 1))

        # Test 1 - Critic's performance on real images
        pred_real = self.critic(x_real)
        
        # Test 2 - Critic's performance on synthetic images
        x_synth = self.synthesize_images(batch_size, training=False)
        pred_synth = self.critic(x_synth)

        # Update the metrics, configured in 'compile()'.
        self.accuracy_real_metric.update_state(y_true=y_real, y_pred=pred_real)
        self.accuracy_synth_metric.update_state(y_true=y_synth, y_pred=pred_synth)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

    def get_config(self):
        config = super(WGAN, self).get_config()
        config.update({
            'critic_class':self.critic.__class__,
            'critic': self.critic.get_config(),
        })
        for key in ['discriminator', 'discriminator_class']:
            config.pop(key, None)
        return config

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        config.update({
            'generator':config['generator_class'].from_config(config['generator']),
            'critic':config['critic_class'].from_config(config['critic'])
        })
        for key in ['generator_class', 'critic_class']:
            config.pop(key, None)
        return keras.Model.from_config(config, custom_objects)

if __name__ == '__main__':
    from dataloader import dataloader
    from models.GANs.utils import MakeSyntheticGIFCallback

    ds = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        batch_size_train=128,
        batch_size_test=1000,
        drop_remainder=True
    )

    generator = define_generator(name='WGenerator')
    critic = define_critic(name='WCritic')
    wgan = WGAN(
        generator=generator,
        critic=critic,
        latent_dim=100,
        image_dim=[28, 28, 1]
    )
    wgan.build()
    wgan.summary(with_graph=True, line_length=120, expand_nested=True)
    wgan.compile(n_critic=5)

    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{wgan.name}_{wgan.generator.name}_{wgan.critic.name}.csv',
        append=True
    )
    gif_maker = MakeSyntheticGIFCallback(
        f'./logs/{wgan.name}_{wgan.generator.name}_{wgan.critic.name}.gif',
        nrows=5, ncols=5,
        postprocess_fn=lambda x:(x+1)/2
    )
    wgan.evaluate(ds['test'])
    wgan.fit(
        x=ds['train'],
        epochs=50,
        verbose=1,
        callbacks=[csv_logger, gif_maker],
        validation_data=ds['test'],
    )