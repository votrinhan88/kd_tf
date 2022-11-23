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

    def compile(self,
                optimizer_disc:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                optimizer_gen:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
                main_loss_fn:keras.losses.Loss=keras.losses.BinaryCrossentropy(),        
                # Sparse CE for normal label, Categorial CE for one-hot label
                aux_loss_fn:keras.losses.Loss=keras.losses.CategoricalCrossentropy(),
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
            loss_fn=None,
            **kwargs
        )
        self.main_loss_fn = main_loss_fn
        self.aux_loss_fn = aux_loss_fn
        self.loss_aux_metric = keras.metrics.Mean(name='loss_aux')
        self.accuracy_aux_metric = keras.metrics.CategoricalCrossentropy(name='accuracy_aux')

    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:
        train_metrics = super(ACGAN, self).train_metrics
        train_metrics.append(
            self.loss_aux_metric,
            self.accuracy_aux_metric
        )
        return train_metrics
        # return [self.loss_real_metric, self.loss_synth_metric, self.loss_gen_metric]
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        test_metrics = super(ACGAN, self).val_metrics
        test_metrics.append(
            self.loss_aux_metric,
            self.accuracy_aux_metric
        )
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

    def train_step(self, data):
        # Unpack data
        x_real, _ = data
        batch_size = tf.shape(x_real)[0]
        
        self.train_discriminator(x_real, batch_size)
        self.train_generator(batch_size)

        results = {m.name: m.result() for m in self.train_metrics}
        return results
                
    def test_step(self, data):
        return super().test_step(data)

    def get_config(self):
        config = super(GAN, self).get_config()
        config.update({
        })
        return config

    @classmethod
    def from_config(cls, config:dict, custom_objects=None):
        # config.update({
        #     'generator':config['generator_class'].from_config(config['generator']),
        #     'discriminator':config['discriminator_class'].from_config(config['discriminator'])
        # })
        # for key in ['generator_class', 'discriminator_class']:
        #     config.pop(key, None)
        return super(GAN, cls).from_config(config, custom_objects)

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
                self.aux_pred = keras.layers.Dense(units=self.num_classes, name='aux_pred', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.aux_pred = keras.layers.Dense(units=self.num_classes, name='aux_pred', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=1, use_bias=False, name='logits')
            self.aux_logits = keras.layers.Dense(units=self.num_classes, name='aux_logits')

    def call(self, inputs, training:bool=False):
        x = inputs
        for block in self.conv_block:
            x = block(x, training=training)
        x = self.flatten(x)
        if self.return_logits is False:
            main_branch = self.pred(x)
            aux_branch = self.aux_pred(x)
        elif self.return_logits is True:
            main_branch = self.logits(x)
            aux_branch = self.aux_logits(x)
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
    
