# Auxiliary Classifier Generative Adversarial Network

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

def keras_example(latent_dim:int=100):
    # Fixed for mnist
    IMAGE_DIM = [28, 28, 1]
    NUM_CLASSES = 10

    def define_example_generator(latent_dim:int):
        latent_input = keras.layers.Input(shape=[latent_dim])

        label_input = keras.layers.Input(shape=[1], dtype='int32')
        label_branch = keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=latent_dim, input_length=1, embeddings_initializer='glorot_normal')(label_input)
        label_branch = keras.layers.Flatten()(label_branch)

        # Concatenate by element-wise product (interesting!)
        main_branch = keras.layers.multiply([latent_input, label_branch])
        main_branch = keras.layers.Dense(3 * 3 * 384, activation='relu')(main_branch)
        main_branch = keras.layers.Reshape((3, 3, 384))(main_branch)
        # upsample to (7, 7, ...)
        main_branch = keras.layers.Conv2DTranspose(192, 5, strides=1, padding='valid', activation='relu', kernel_initializer='glorot_normal')(main_branch)
        main_branch = keras.layers.BatchNormalization()(main_branch)
        # upsample to (14, 14, ...)
        main_branch = keras.layers.Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal')(main_branch)
        main_branch = keras.layers.BatchNormalization()(main_branch)
        # upsample to (28, 28, ...)
        outputs = keras.layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh', kernel_initializer='glorot_normal')(main_branch)
        
        generator = keras.Model([latent_input, label_input], outputs)
        generator.build(input_shape=[[None, latent_dim], [None, 1]])
        return generator

    def define_example_discriminator():
        image_input = keras.layers.Input(shape=IMAGE_DIM)
        main_branch = keras.layers.Conv2D(32, 3, padding='same', strides=2, input_shape=(28, 28, 1))(image_input)
        main_branch = keras.layers.LeakyReLU(0.2)(main_branch)
        main_branch = keras.layers.Dropout(0.3)(main_branch)
        main_branch = keras.layers.Conv2D(64, 3, padding='same', strides=1)(main_branch)
        main_branch = keras.layers.LeakyReLU(0.2)(main_branch)
        main_branch = keras.layers.Dropout(0.3)(main_branch)
        main_branch = keras.layers.Conv2D(128, 3, padding='same', strides=2)(main_branch)
        main_branch = keras.layers.LeakyReLU(0.2)(main_branch)
        main_branch = keras.layers.Dropout(0.3)(main_branch)
        main_branch = keras.layers.Conv2D(256, 3, padding='same', strides=1)(main_branch)
        main_branch = keras.layers.LeakyReLU(0.2)(main_branch)
        main_branch = keras.layers.Dropout(0.3)(main_branch)
        main_branch = keras.layers.Flatten()(main_branch)
        
        main_output = keras.layers.Dense(1, activation='sigmoid', name='generation')(main_branch)
        aux_output = keras.layers.Dense(NUM_CLASSES, activation='softmax', name='auxiliary')(main_branch)
        
        return keras.Model(image_input, [main_output, aux_output])

    gen = define_example_generator(latent_dim=latent_dim)
    # gen.summary()

    disc = define_example_discriminator()
    # disc.summary()

# TODO: Update train_step, test_step
class ACGAN(GAN):
    _name = 'ACGAN'
    def __init__(self,
                 generator:keras.Model,
                 discriminator:keras.Model,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 num_classes:Union[None, int]=None,
                 onehot_input:Union[None, bool]=None,
                 **kwargs):
        """Initialize ACGAN."""
        # The same method to CGAN
        super(ACGAN, self).__init__(
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
        pred, pred_aux = self.discriminator.call(x_synth, training=training)
        return pred, pred_aux

    def compile(self,
                optimizer_disc:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                optimizer_gen:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                loss_fn:keras.losses.Loss=keras.losses.BinaryCrossentropy(),        
                # Sparse CE for normal label, Categorial CE for one-hot label
                loss_fn_aux:keras.losses.Loss=keras.losses.CategoricalCrossentropy(),
                **kwargs):
        """Compile GAN.
        
        Args:
            `optimizer_disc`: Optimizer for discriminator.
                Defaults to `keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)`.
            `optimizer_gen`: Optimizer for generator.
                Defaults to `keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)`.
            `main_loss_fn`: Loss function for main output.
                Defaults to `keras.losses.BinaryCrossentropy()`.
            `main_loss_fn`: Loss function for auxiliary output.
        """                
        super(ACGAN, self).compile(
            optimizer_disc=optimizer_disc,
            optimizer_gen=optimizer_gen,
            loss_fn=loss_fn,
            **kwargs)
        self.loss_fn_aux = loss_fn_aux

        # Additional metrics
        self.accuracy_aux_real_metric = keras.metrics.CategoricalCrossentropy(name='accuracy_aux_real')
        self.accuracy_aux_synth_metric = keras.metrics.CategoricalCrossentropy(name='accuracy_aux_synth')
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        test_metrics = super(ACGAN, self).val_metrics
        test_metrics.append(self.accuracy_aux_real_metric)
        test_metrics.append(self.accuracy_aux_synth_metric)
        return test_metrics

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
    
    def synthesize_images(self, label, batch_size:int, training:bool=False):
        latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
        x_synth = self.generator([latent_noise, label], training=training)
        return x_synth

    def train_discriminator(self, x_real, label, batch_size:int):
        # Phase 1 - Training the discriminator
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.discriminator.trainable_variables)
    
            x_synth = self.synthesize_images(label=label, batch_size=batch_size, training=True)

            pred_real, pred_aux_real = self.discriminator(x_real, training=True)
            pred_synth, pred_aux_synth = self.discriminator(x_synth, training=True)

            loss_real = self.loss_fn(y_real, pred_real) + self.loss_fn_aux(label, pred_aux_real)
            loss_synth = self.loss_fn(y_synth, pred_synth) + self.loss_fn_aux(label, pred_aux_synth)
        # Back-propagation
        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(loss_real, trainable_vars)        
        self.optimizer_disc.apply_gradients(zip(gradients, trainable_vars))
        gradients = tape.gradient(loss_synth, trainable_vars)        
        self.optimizer_disc.apply_gradients(zip(gradients, trainable_vars))
        del tape

        self.loss_real_metric.update_state(loss_real)
        self.loss_synth_metric.update_state(loss_synth)

    def train_generator(self, label, batch_size:int):
        # Phase 2 - Training the generator
        y_real = tf.ones(shape=(batch_size, 1))

        # if label is None:
        label = tf.one_hot(
            indices=tf.random.uniform(
                shape=(batch_size),
                minval=0, maxval=self.num_classes, dtype=tf.int32),
            depth=self.num_classes,
            axis=1
        )

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.generator.trainable_variables)
            x_synth = self.synthesize_images(label=label, batch_size=batch_size, training=True)
            pred_synth, pred_aux_synth = self.discriminator(x_synth, training=True)
            loss_gen = self.loss_fn(y_real, pred_synth) + self.loss_fn_aux(label, pred_aux_synth)
        # Back-propagation
        trainable_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_gen, trainable_vars)        
        self.optimizer_gen.apply_gradients(zip(gradients, trainable_vars))
        del tape

        # Update the metrics, configured in 'compile()'.
        self.loss_gen_metric.update_state(loss_gen)

    def train_step(self, data):
        # Unpack data
        x_real, label = data
        batch_size = tf.shape(data[0])[0]
        
        self.train_discriminator(x_real, label, batch_size)
        self.train_generator(label, batch_size)

        results = {m.name: m.result() for m in self.train_metrics}
        return results
                
    def test_step(self, data):
        # Unpack data
        x_real, label = data
        batch_size:int = x_real.shape[0]
        y_synth = tf.zeros(shape=(batch_size, 1))
        y_real = tf.ones(shape=(batch_size, 1))

        # Test 1 - Discriminator's performance on real images
        pred_real, pred_aux_real = self.discriminator(x_real)
        
        # Test 2 - Discriminator's performance on synthetic images
        x_synth = self.synthesize_images(label=label, batch_size=batch_size, training=False)
        pred_synth, pred_aux_synth = self.discriminator(x_synth)

        # Update the metrics, configured in 'compile()'.
        self.accuracy_real_metric.update_state(y_true=y_real, y_pred=pred_real)
        self.accuracy_synth_metric.update_state(y_true=y_synth, y_pred=pred_synth)
        self.accuracy_aux_real_metric.update_state(y_true=label, y_pred=pred_aux_real)
        self.accuracy_aux_synth_metric.update_state(y_true=label, y_pred=pred_aux_synth)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

    def get_config(self):
        config = super(ACGAN, self).get_config()
        config.update({
            'generator_class':self.generator.__class__,
            'generator': self.generator.get_config(),
            'discriminator_class':self.discriminator.__class__,
            'discriminator': self.discriminator.get_config(),
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim,
            'num_classes': self.num_classes,
            'onehot_input': self.onehot_input,
        })
        return config

class AC_Discriminator(keras.Model):
    _name = 'ACDisc'
    
    """Discriminator for ACGAN. Based on architecture of DC_Discriminator."""
    _name = 'DCDisc'

    def __init__(self,
                 image_dim:List[int]=[28, 28, 1],
                 base_dim:List[int]=[7, 7, 256],
                 num_classes:int=10,
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
        self.num_classes = num_classes
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
            if self.num_classes == 1:
                self.pred_aux = keras.layers.Dense(units=self.num_classes, name='pred_aux', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.pred_aux = keras.layers.Dense(units=self.num_classes, name='pred_aux', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, use_bias=False, name='logits')
            self.logits_aux = keras.layers.Dense(units=self.num_classes, name='logits_aux')

    def call(self, inputs, training:bool=False):
        x = inputs
        for block in self.conv_block:
            x = block(x, training=training)
        x = self.flatten(x)
        if self.return_logits is False:
            main_branch = self.pred(x)
            aux_branch = self.pred_aux(x)
        elif self.return_logits is True:
            main_branch = self.logits(x)
            aux_branch = self.logits_aux(x)
        return main_branch, aux_branch

    def get_config(self):
        config = super(AC_Discriminator, self).get_config()
        config.update({
            'image_dim': self.image_dim,
            'base_dim': self.base_dim,
            'num_classes': self.num_classes,
            'return_logits': self.return_logits,            
        })
        return config

    def build(self):
        super().build(input_shape=[None, *self.image_dim])

    def summary(self, with_graph:bool=False, **kwargs):
        inputs = keras.layers.Input(shape=self.image_dim)
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

if __name__ == '__main__':
    # keras_example()
    from dataloader import dataloader
    from models.GANs.CGAN import ConditionalGeneratorEmbed
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback

    ds, info = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        batch_size_train=64,
        batch_size_test=1000,
        drop_remainder=True,
        onehot_label=True,
        with_info=True)
    class_names = info.features['label'].names

    gen = ConditionalGeneratorEmbed(
        latent_dim=100,
        image_dim=[28, 28, 1],
        base_dim=[7, 7, 256],
        embed_dim=50,
        num_classes=10,
        onehot_input=True
    )
    gen.build()

    disc = AC_Discriminator(
        image_dim=[28, 28, 1],
        base_dim=[7, 7, 256],
        num_classes=10
    )
    disc.build()

    acgan = ACGAN(generator=gen, discriminator=disc)
    acgan.build()
    acgan.summary(with_graph=True, expand_nested=True, line_length=120)
    acgan.compile(
        optimizer_disc=keras.optimizers.Adam(learning_rate=2e-4),   
        optimizer_gen=keras.optimizers.Adam(learning_rate=2e-4),
        loss_fn=keras.losses.BinaryCrossentropy(),
        # loss_fn_aux=keras.losses.CategoricalCrossentropy(),
        loss_fn_aux=keras.losses.KLDivergence(),
    )
    
    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{acgan.name}_{acgan.generator.name}_{acgan.discriminator.name}.csv',
        append=True)
    
    gif_maker = MakeConditionalSyntheticGIFCallback(
        filename=f'./logs/{acgan.name}_{acgan.generator.name}_{acgan.discriminator.name}.gif', 
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    slerper = MakeInterpolateSyntheticGIFCallback(
        filename=f'./logs/{acgan.name}_{acgan.generator.name}_{acgan.discriminator.name}_itpl_slerp.gif',
        itpl_method='slerp',
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    acgan.fit(
        x=ds['train'],
        epochs=50,
        callbacks=[csv_logger, gif_maker, slerper],
        validation_data=ds['test'],
    )