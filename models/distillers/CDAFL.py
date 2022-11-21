import warnings
from typing import List, Union, Tuple
import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.distillers.DataFreeDistiller import DataFreeDistiller, DataFreeGenerator
else:
    from .DataFreeDistiller import DataFreeDistiller, DataFreeGenerator

class ConditionalDataFreeGenerator(DataFreeGenerator):
    """Review GAN to CGAN to modify.
    """
    _name = 'CDAFL_Gen'
    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[32, 32, 3],
                 embed_dim:int=50,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 dafl_batchnorm:bool=True,
                 **kwargs):
        assert isinstance(dafl_batchnorm, bool), '`dafl_batchnorm` must be of type bool'
        assert isinstance(onehot_input, bool), '`onehot_input` must be of type bool.'

        keras.Model.__init__(self, name=self._name, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.dafl_batchnorm = dafl_batchnorm

        self._BASE_DIM = [self.image_dim[0]//4, self.image_dim[1]//4]
        if self.dafl_batchnorm is True:
            self._EPSILONS = [1e-5, 0.8, 0.8, 1e-5]
            self._MOMENTUM = 0.9
        elif self.dafl_batchnorm is False:
            self._EPSILONS = [keras.layers.BatchNormalization().epsilon]*4 # 1e-3
            self._MOMENTUM = keras.layers.BatchNormalization().momentum # 0.99

        # Latent branch, converted from DAFL's generator
        self.latent_branch = keras.Sequential([
            keras.layers.Dense(units=tf.math.reduce_prod(self._BASE_DIM)*128, use_bias=False),
            keras.layers.Reshape(target_shape=[*self._BASE_DIM, 128]),
            keras.layers.ReLU()
        ])

        # Conditional label branch
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.label_branch = keras.Sequential([
            keras.layers.Dense(units=self.embed_dim),
            keras.layers.ReLU(),
            keras.layers.Dense(units=tf.math.reduce_prod(self._BASE_DIM)),
            keras.layers.Reshape(target_shape=(*self._BASE_DIM, 1)),
            keras.layers.ReLU(),
        ])

        # Main branch: concat both branches and upsample
        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        self.concat = keras.layers.Concatenate()

        self.conv_block_0 = keras.Sequential(
            layers=[keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[0])],
            name='conv_block_0'
        )

        self.upsamp_1 = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='upsamp_1')
        self.conv_block_1 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[1]),
                keras.layers.LeakyReLU(alpha=0.2)
            ],
            name='conv_block_1'
        )
        
        self.upsamp_2 = keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest", name='upsamp_2')
        self.conv_block_2 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[2]),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(filters=self.image_dim[2], kernel_size=3, strides=1, padding='same'),
                keras.layers.Activation(tf.nn.tanh),
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[3], center=False, scale=False)],
            name='conv_block_2'
        )

    def call(self, inputs, training:bool=False):
        # Parse inputs
        latents, labels = inputs
        # Forward
        latents = self.latent_branch(latents)
        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        labels = self.label_branch(labels)
        x = self.concat([latents, labels])
        x = self.conv_block_0(x, training=training)
        x = self.upsamp_1(x)
        x = self.conv_block_1(x, training=training)
        x = self.upsamp_2(x)
        x = self.conv_block_2(x, training=training)
        return x
    
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'latent_dim': self.latent_dim,
            'image_dim': self.image_dim,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'onehot_input': self.onehot_input,
            'dafl_batchnorm': self.dafl_batchnorm,            
        })
        return config

class CDAFL(DataFreeDistiller):
    """Actually does not need to write from scratch a new class, only needs to write a
    new method for generating synthetic images.

    Originally in train_step():
        latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
        x_synth = self.generator(latent_noise, training=True)
        --> Can we wrap this in a method, and overwrite later?

    So for CGAN, should add like this in train_step():
        latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
        label = <randomly evenly-distributed labels>
        x_synth = self.generator([latent_noise, label], training=True)

        ...
        # Teacher predicts one of num_classes
        teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)
        # But we don't have to use pseudo-label anymore
        # --> One-hot loss can be changed to KLDiv/CE loss.
        ...
        # Student also predicts one of num_classes
        student_prob = self.student(x_synth, training=True)
        ...
        # Distillation is still the same
        loss_distill = self.distill_loss_fn(teacher_prob, student_prob)
    """
    _name = 'ConditionalDataFreeDistiller'
    def __init__(self,
                 teacher:keras.Model,
                 student:keras.Model,
                 generator:DataFreeGenerator,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 embed_dim:Union[None, int]=None,
                 num_classes:Union[None, int]=None,
                 onehot_input:Union[None, bool]=None,
                 **kwargs):
        super(CDAFL, self).__init__(
            teacher=teacher,
            student=student,
            generator=generator,
            latent_dim=latent_dim,
            image_dim=image_dim,
            **kwargs)

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

    def compile(self,
                optimizer_student:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8),
                optimizer_generator:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8),
                onehot_loss_fn:Union[bool, keras.losses.Loss]=True,
                activation_loss_fn:Union[bool, keras.losses.Loss]=True,
                info_entropy_loss_fn:Union[bool, keras.losses.Loss]=True,
                distill_loss_fn:keras.losses.Loss=keras.losses.KLDivergence(),
                student_loss_fn:keras.losses.Loss=keras.losses.SparseCategoricalCrossentropy(),
                batch_size:int=500,
                num_batches:int=120,
                alpha:float=0.1,
                beta:float=5,
                confidence:float=None,
                **kwargs):
        if batch_size % self.num_classes > 0:
            warnings.warn(
                f'`batch_size` {batch_size} is not divisible by `num_classes` '+
                f'{self.num_classes} and will give unevenly distributed batches.')

        super(CDAFL, self).compile(
            optimizer_student=optimizer_student,
            optimizer_generator=optimizer_generator,
            onehot_loss_fn=onehot_loss_fn,
            activation_loss_fn=activation_loss_fn,
            info_entropy_loss_fn=info_entropy_loss_fn,
            distill_loss_fn=distill_loss_fn,
            student_loss_fn=student_loss_fn,
            batch_size=batch_size,
            num_batches=num_batches,
            alpha=alpha,
            beta=beta,
            confidence=confidence,
            **kwargs)
        
        # # Re-config one-hot loss
        # if self.onehot_loss_fn is True:
        #     if self.onehot_input is True:
        #         # KLDiv should also work
        #         self._onehot_loss_fn = keras.losses.CategoricalCrossentropy()
        #     elif self.onehot_input is False:
        #         self._onehot_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        # elif self.onehot_loss_fn is False:
        #     self._onehot_loss_fn = lambda *args, **kwargs:0
        # else:
        #     self._onehot_loss_fn = self.onehot_loss_fn
        
    def train_step(self, data):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            # Specify trainable variables
            tape.watch(self.generator.trainable_variables)
            tape.watch(self.student.trainable_variables)

            # Phase 1 - Training the conditional Generator
            latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            label = tf.cast(
                tf.math.floor(tf.linspace(start=0, stop=self.num_classes, num=self.batch_size+1)[0:-1]),
                dtype=tf.int32
            )
            if self.onehot_input is True:
                oh_label = tf.one_hot(indices=label, depth=self.num_classes)
                x_synth = self.generator([latent_noise, oh_label], training=True)
            elif self.onehot_input is False:
                x_synth = self.generator([latent_noise, label], training=True)
            teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)

            loss_onehot = self._onehot_loss_fn(label, teacher_prob)
            loss_activation = self._activation_loss_fn(teacher_fmap)
            loss_info_entropy = self._info_entropy_loss_fn(teacher_prob)
            loss_generator = loss_onehot + self.alpha*loss_activation + self.beta*loss_info_entropy
            
            # Phase 2: Training the student network.
            # Detach gradient graph of generator and teacher
            x_synth = tf.stop_gradient(tf.identity(x_synth))
            teacher_prob = tf.stop_gradient(tf.identity(teacher_prob))

            if self.confidence is not None:
                # Keep only images with high confidence to train student
                confident_idx = tf.squeeze(tf.where(tf.math.reduce_max(teacher_prob, axis=1) >= self.confidence), axis=1)
                x_synth = tf.gather(params=x_synth, indices=confident_idx)
                teacher_prob = tf.gather(params=teacher_prob, indices=confident_idx)

            student_prob = self.student(x_synth, training=True)
            loss_distill = self.distill_loss_fn(teacher_prob, student_prob)

        # Back-propagation of Generator
        generator_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_generator, generator_vars)
        self.optimizer_generator.apply_gradients(zip(gradients, generator_vars))
        # Back-propagation of Student
        student_vars = self.student.trainable_variables
        gradients = tape.gradient(loss_distill, student_vars)
        self.optimizer_student.apply_gradients(zip(gradients, student_vars))
        del tape

        # Update the metrics, configured in 'compile()'
        self.loss_onehot_metric.update_state(loss_onehot)
        self.loss_activation_metric.update_state(loss_activation)
        self.loss_info_entropy_metric.update_state(loss_info_entropy)
        self.loss_generator_metric.update_state(loss_generator)
        self.loss_distill_metric.update_state(loss_distill)
        results = {m.name: m.result() for m in self.train_metrics}
        return results

if __name__ == '__main__':
    from dataloader import dataloader
    from models.classifiers.LeNet_5 import LeNet_5_ReLU_MaxPool
    from models.distillers.utils import add_fmap_output
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback

    tf.config.run_functions_eagerly(True)

    def run_experiment_mnist(pretrained_teacher:bool=False):
        # Experiment 4.1: Classification result on the MNIST dataset
        #                                       LeNet-5        HintonNets
        # Teacher:                              LeNet-5        Hinton-784-1200-1200-10
        # Student/Baseline:                     LeNet-5-HALF   Hinton-784-800-800-10
        # Teacher:                              98.91%         98.39%
        # Baseline:                             98.65%         98.11%
        # Traditional KD:                       98.91%         98.39%
        # KD with randomly generated noise:     88.01%         87.58%
        # KD with meta-data KD:                 92.47%         91.24%
        # KD with alternative dataset USPS:     94.56%         93.99%
        # Data-free KD:                         98.20%         97.91%    
        print(' Experiment 4.1: Classification result on the MNIST dataset '.center(80,'#'))

        ds, info = dataloader(
            dataset='mnist',
            resize=[32, 32],
            rescale='standardization',
            batch_size_train=256,
            batch_size_test=1024,
            with_info=True
        )
        class_names = info.features['label'].names

        # Pretrained teacher
        teacher = LeNet_5_ReLU_MaxPool()
        teacher.compile(
            metrics=['accuracy'], 
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy())
        teacher.build()

        if pretrained_teacher is True:
            teacher.load_weights('./pretrained/mnist/LeNet-5_ReLU_MaxPool_9908.h5')
        elif pretrained_teacher is False:
            best_callback = keras.callbacks.ModelCheckpoint(
                filepath=f'./logs/{teacher.name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
            )
            csv_logger = keras.callbacks.CSVLogger(
                filename=f'./logs/{teacher.name}.csv',
                append=True
            )
            teacher.fit(
                ds['train'],
                epochs=10,
                callbacks=[best_callback, csv_logger],
                shuffle=True,
                validation_data=ds['test']
            )
            teacher.load_weights(filepath=f'./logs/{teacher.name}_best.h5')
        teacher.evaluate(ds['test'])
        teacher = add_fmap_output(model=teacher, fmap_layer='flatten')

        # Student (LeNet-5-HALF)
        student = LeNet_5_ReLU_MaxPool(half=True)
        student.build()
        student.compile(metrics='accuracy')
        student.evaluate(ds['test'])

        generator = ConditionalDataFreeGenerator(
            latent_dim=100,
            image_dim=[32, 32, 1],
            embed_dim=50,
            num_classes=10,
            onehot_input=False,
            dafl_batchnorm=True
        )
        generator.build()
        generator.summary(with_graph=True, line_length=120, expand_nested=True)

        # Train one student with default data-free learning settings
        distiller = CDAFL(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8),
            optimizer_generator=keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8),
            onehot_loss_fn=True,
            activation_loss_fn=True,
            info_entropy_loss_fn=True,
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=500,
            num_batches=120,
            alpha=0.1,
            beta=5,
            confidence=None
        )

        csv_logger = keras.callbacks.CSVLogger(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.gif',
            postprocess_fn=lambda x:x*0.5 + 0.5, 
            class_names=class_names,
            delete_png=True
        )
        slerper = MakeInterpolateSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_itpl_slerp.gif',
            itpl_method='slerp',
            postprocess_fn=lambda x:x*0.5 + 0.5, 
            class_names=class_names
        )
        distiller.fit(
            epochs=200,
            # callbacks=[csv_logger, gif_maker, slerper],
            callbacks=[csv_logger, gif_maker],
            validation_data=ds['test']
        )

    run_experiment_mnist()