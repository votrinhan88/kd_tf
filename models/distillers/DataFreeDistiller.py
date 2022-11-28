import warnings
from typing import List, Union
import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.distillers.utils import PlaceholderDataGenerator, LearningRateSchedulerCustom
else:
    from .utils import PlaceholderDataGenerator, LearningRateSchedulerCustom

class DataFreeGenerator(keras.Model):
    """DCGAN Generator model implemented in Data-Free Learning of Student Networks - 
    Chen et al. (2019), replicated with the same architecture.

    Original: Unsupervised representation learning with deep convolutional
    generative adversarial networks - Radford et al. (2015)
    DOI: 10.48550/arXiv.1511.06434

    Args:
        `latent_dim`: Dimension of latent space. Defaults to `100`.
        `image_dim`: Dimension of synthetic images. Defaults to `[32, 32, 3]`.
        `dafl_batchnorm`: Flag to use same configuration for Batch Normalization
            layers as in original DAFL paper. Defaults to `True`.
    """    
    _name = 'DataFreeGen'

    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[32, 32, 3],
                 dafl_batchnorm:bool=True,
                 **kwargs):
        """Initialize generator.
        
        Args:
            `latent_dim`: Dimension of latent space. Defaults to `100`.
            `image_dim`: Dimension of synthetic images. Defaults to `[32, 32, 3]`.
            `dafl_batchnorm`: Flag to use same configuration for Batch Normalization
                layers as in original DAFL paper. Defaults to `True`.
        """
        assert isinstance(dafl_batchnorm, bool), '`dafl_batchnorm` must be of type bool'
        super().__init__(self, name=self._name, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.dafl_batchnorm = dafl_batchnorm

        self._INIT_DIM = [self.image_dim[0]//4, self.image_dim[1]//4]
        if self.dafl_batchnorm is True:
            self._EPSILONS = [1e-5, 0.8, 0.8, 1e-5]
            self._MOMENTUM = 0.9
        elif self.dafl_batchnorm is False:
            self._EPSILONS = [keras.layers.BatchNormalization().epsilon]*4 # 1e-3
            self._MOMENTUM = keras.layers.BatchNormalization().momentum # 0.99

        self.dense = keras.layers.Dense(units=self._INIT_DIM[0] * self._INIT_DIM[1] * 128)
        self.reshape = keras.layers.Reshape(target_shape=(self._INIT_DIM[0], self._INIT_DIM[1], 128))
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
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_block_0(x, training=training)
        x = self.upsamp_1(x)
        x = self.conv_block_1(x, training=training)
        x = self.upsamp_2(x)
        x = self.conv_block_2(x, training=training)
        return x

    def build(self):
        super(DataFreeGenerator, self).build(input_shape=[None, self.latent_dim])

    def summary(self, with_graph:bool=False, **kwargs):
        inputs = keras.layers.Input(shape=[self.latent_dim])
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
            'dafl_batchnorm': self.dafl_batchnorm
        })
        return config

# TODO: update `train_batch_exact`
# Don't use attributes latent_dim and image_dim directly from generator in later calls
class DataFreeDistiller(keras.Model):
    """A knowledge distillation scheme performed without the training set and
    architecture information of the teacher model, utilizing a generator
    approximating the original dataset.
    
    Args:
        `teacher`: Pre-trained teacher model.
        `student`: To-be-trained student model.
        `generator`: DCGAN generator proposed in study.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            generator. Defaults to `None`.
        `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
            generator. Defaults to `None`.
    
    Data-Free Learning of Student Networks - Chen et al. (2019)         
    DOI: 10.48550/arXiv.1904.01186  
    
    Implementation in PyTorch: https://github.com/autogyro/DAFL
    """
    _name = 'DataFreeDistiller'

    def __init__(self,
                 teacher:keras.Model,
                 student:keras.Model,
                 generator:DataFreeGenerator,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 **kwargs):
        """Initialize distiller.
        
        Args:
            `teacher`: Pre-trained teacher model.
            `student`: To-be-trained student model.
            `generator`: DCGAN generator proposed in study.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                generator. Defaults to `None`.
            `image_dim`: Dimension of synthetic image, leave as `None` to be parsed from
                generator. Defaults to `None`.
        """
        super().__init__(name=self._name, **kwargs)
        self.teacher = teacher
        self.student = student
        self.generator = generator

        if latent_dim is None:
            self.latent_dim:int = self.generator.latent_dim
        elif latent_dim is not None:
            self.latent_dim = latent_dim

        if image_dim is None:
            self.image_dim:int = self.generator.image_dim
        elif image_dim is not None:
            self.image_dim = image_dim

    def compile(self,
                optimizer_student:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8),
                optimizer_generator:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8),
                onehot_loss_fn:Union[bool, keras.losses.Loss]=True,
                activation_loss_fn:Union[bool, keras.losses.Loss]=True,
                info_entropy_loss_fn:Union[bool, keras.losses.Loss]=True,
                distill_loss_fn:keras.losses.Loss=keras.losses.KLDivergence(),
                student_loss_fn:keras.losses.Loss=keras.losses.SparseCategoricalCrossentropy(),
                batch_size:int=512,
                num_batches:int=120,
                coeff_oh:float=1,
                coeff_ac:float=0.1,
                coeff_ie:float=5,
                confidence:float=None,
                **kwargs):
        """Compile distiller.
        
        Args:
            `optimizer_student`: Optimizer for student model.
                Defaults to `keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8)`.
            `optimizer_generator`: .
                Defaults to `keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8)`.
            `onehot_loss_fn`: One-hot loss function, as in original paper.
                Options:
                    `True`: use default `keras.losses.SparseCategoricalCrossentropy()`
                    `False`: toggle off
                    Others: custom user-defined loss function.
                Defaults to `True`.
            `activation_loss_fn`: Activation loss function, as in original paper.
                Options:
                    `True`: Use default (see `_activation_loss_fn`)
                    `False`: Toggle off
                    Others: Custom user-defined loss function
                Defaults to `True`.
            `info_entropy_loss_fn`: Information entropy loss function, as in original
            paper.
                Options:
                    `True`: Use default (see `_info_entropy_loss_fn`)
                    `False`: Toggle off
                    Others: Custom user-defined loss function
                Defaults to `True`.
            `distill_loss_fn`: Distillation loss function.
                Defaults to `keras.losses.KLDivergence()`.
            `student_loss_fn`: Loss function to evaluate the student's performance on
            the validation set.
                Defaults to `keras.losses.SparseCategoricalCrossentropy()`.
            `batch_size`: Size of each synthetic batch. Defaults to `512`.
            `num_batches`: Number of training batches each epoch. Defaults to `120`.
            `coeff_ac`: Coefficient of activation loss. Defaults to `0.1`.
            `beta`: Coefficient of information entropy loss. Defaults to `5`.
            `confidence`: Confidence threshold for filtering out low-quality synthetic
            images (evaluated by the teacher) before distillation.
                Options:
                    `None`: do not apply
                    `float` number in the range [0, 1]: apply with one threshold
                Defaults to `None`.
        """

        if not isinstance(onehot_loss_fn, (keras.losses.Loss, bool)):
            warnings.warn('`onehot_loss_fn` should be of type `keras.losses.Loss` or `bool`.')
        if not isinstance(activation_loss_fn, (keras.losses.Loss, bool)):
            warnings.warn('`activation_loss_fn` should be of type `keras.losses.Loss` or `bool`.')
        if not isinstance(info_entropy_loss_fn, (keras.losses.Loss, bool)):
            warnings.warn('`info_entropy_loss_fn` should be of type `keras.losses.Loss` or `bool`.')

        super().compile(**kwargs)
        self.optimizer_student = optimizer_student
        self.optimizer_generator = optimizer_generator
        self.onehot_loss_fn = onehot_loss_fn
        self.activation_loss_fn = activation_loss_fn
        self.info_entropy_loss_fn = info_entropy_loss_fn
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.coeff_oh = coeff_oh
        self.coeff_ac = coeff_ac
        self.coeff_ie = coeff_ie
        self.confidence = confidence

        # Config one-hot loss
        if self.onehot_loss_fn is True:
            self._onehot_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        elif self.onehot_loss_fn is False:
            self._onehot_loss_fn = lambda *args, **kwargs:0
        else:
            self._onehot_loss_fn = self.onehot_loss_fn
        # Config activation loss
        if self.activation_loss_fn is True:
            pass
        elif self.activation_loss_fn is False:
            self._activation_loss_fn = lambda *args, **kwargs:0
        else:
            self._activation_loss_fn = self.activation_loss_fn
        # Config information entropy loss
        if self.info_entropy_loss_fn is True:
            pass
        elif self.info_entropy_loss_fn is False:
            self._info_entropy_loss_fn = lambda *args, **kwargs:0
        else:
            self._info_entropy_loss_fn = self.info_entropy_loss_fn

        # Placeholder data generator
        self.train_data = PlaceholderDataGenerator(
            num_batches=self.num_batches,
            batch_size=self.batch_size
        )

        # Metrics
        self.loss_onehot_metric = keras.metrics.Mean(name='loss_oh')
        self.loss_activation_metric = keras.metrics.Mean(name='loss_ac')
        self.loss_info_entropy_metric = keras.metrics.Mean(name='loss_ie')
        self.loss_generator_metric = keras.metrics.Mean(name='loss_gen')
        self.loss_distill_metric = keras.metrics.Mean(name='loss_dt')

        self.accuracy_metric = keras.metrics.Accuracy(name='accuracy')
        self.loss_student_metric = keras.metrics.Mean(name='loss_student')

    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:        
        """Metrics monitoring training step.
        
        Returns:
            List of training metrics.
        """        
        return [self.loss_onehot_metric,
                self.loss_activation_metric,
                self.loss_info_entropy_metric,
                self.loss_generator_metric,
                self.loss_distill_metric]
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring validation step.
        
        Returns:
            List of validation metrics.
        """
        return [self.accuracy_metric,
                self.loss_student_metric]

    def train_step(self, data):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            # Specify trainable variables
            tape.watch(self.generator.trainable_variables)
            tape.watch(self.student.trainable_variables)

            # Phase 1: Training the Generator
            latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            x_synth = self.generator(latent_noise, training=True)
            teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)
            pseudo_label = tf.math.argmax(input=teacher_prob, axis=1)

            loss_onehot = self._onehot_loss_fn(pseudo_label, teacher_prob)
            loss_activation = self._activation_loss_fn(teacher_fmap)
            loss_info_entropy = self._info_entropy_loss_fn(teacher_prob)
            loss_generator = self.coeff_oh*loss_onehot + self.coeff_ac*loss_activation + self.coeff_ie*loss_info_entropy
            
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

    def train_batch_exact(self, data):
        # Phase 1: Training the Generator
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            # Specify trainable variables
            tape.watch(self.generator.trainable_variables)

            latent_noise = tf.random.normal(shape=[self.batch_size, self.generator.latent_dim])
            x_synth = self.generator(latent_noise, training=True)
            teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)
            pseudo_label = tf.math.argmax(input=teacher_prob, axis=1)

            loss_onehot = self._onehot_loss_fn(pseudo_label, teacher_prob)
            loss_activation = self._activation_loss_fn(teacher_fmap)
            loss_info_entropy = self._info_entropy_loss_fn(teacher_prob)           
            loss_generator = self.coeff_oh*loss_onehot + self.coeff_ac*loss_activation + self.coeff_ie*loss_info_entropy
        
        # Back-propagation of Generator
        generator_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_generator, generator_vars)
        self.optimizer_generator.apply_gradients(zip(gradients, generator_vars))
        
        # Phase 2: Training the student network.
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            # Specify trainable variables
            tape.watch(self.student.trainable_variables)

            latent_noise = tf.random.normal(shape=[self.batch_size, self.generator.latent_dim])
            x_synth = self.generator(latent_noise, training=True)
            teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)
            pseudo_label = tf.math.argmax(input=teacher_prob, axis=1)

            student_prob = self.student(x_synth, training=True)
            loss_distill = self.distill_loss_fn(teacher_prob, student_prob)

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

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        student_prob = self.student(x, training=False)
        student_pred = tf.math.argmax(input=student_prob, axis=1)

        # Calculate the loss
        loss_student = self.student_loss_fn(y, student_prob)
        
        # Update the metrics, configured in 'compile()'
        self.accuracy_metric.update_state(y_true=y, y_pred=student_pred)
        self.loss_student_metric.update_state(loss_student)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

    def fit(self, **kwargs):
        super().fit(x=self.train_data, steps_per_epoch=self.num_batches, **kwargs)

    @staticmethod
    def _activation_loss_fn(inputs:tf.Tensor) -> tf.Tensor:        
        """Activation loss function. Typical used with the teacher model's
        flattened feature map.
        
        Args:
            `inputs`: teacher model's flattened feature map. 
        Returns:
            Loss value.
        """        
        loss = -tf.math.reduce_mean(tf.abs(inputs), axis=[0, 1])
        return loss

    @staticmethod
    def _info_entropy_loss_fn(inputs):
        """Information entropy loss function. Typically used with the teacher model's
        prediction (probability).
        
        Args:
            `inputs`: teacher model's prediction (probability).
        Returns:
            Loss value.
        """        
        distr_synthetic = tf.math.reduce_mean(inputs, axis=0)
        loss = tf.math.reduce_sum(distr_synthetic*tf.experimental.numpy.log10(distr_synthetic))
        return loss

class DataFreeDistiller_Multiple(keras.Model):
    """Re-implementation of DataFreeDistiller to work with multiple students using
    different confidence thresholds and rebalancing conditions. The students are
    initialized with the same weights and trained with the same optimizer
    configurations.

    Args:
        `teacher`:  Pre-trained teacher model.
        `student`:  To-be-trained student model.
        `generator`:  DCGAN generator proposed in study.
    """
    _name = 'DataFreeDistiller_Multiple'

    def __init__(self,
                 teacher:keras.Model,
                 student:keras.Model,
                 generator:DataFreeGenerator,
                 **kwargs):
        """Initialize distiller.

        Args:
            `teacher`:  Pre-trained teacher model.
            `student`:  To-be-trained student model.
            `generator`:  DCGAN generator proposed in study.
        """
        super().__init__(name=self._name, **kwargs)
        self.teacher = teacher
        self.student = student
        self.generator = generator

    def compile(self,
                optimizer_student:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8),
                optimizer_generator:keras.optimizers.Optimizer=keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8),
                onehot_loss_fn:Union[bool, keras.losses.Loss]=True,
                activation_loss_fn:Union[bool, keras.losses.Loss]=True,
                info_entropy_loss_fn:Union[bool, keras.losses.Loss]=True,
                distill_loss_fn:keras.losses.Loss=keras.losses.KLDivergence(),
                student_loss_fn:keras.losses.Loss=keras.losses.SparseCategoricalCrossentropy(),
                batch_size:int=512,
                num_batches:int=120,
                alpha:float=0.1,
                beta:float=5,
                # temperature:float=1,
                confidence:Union[List[Union[float, None]], float, None]=None,
                rebalance:Union[List[bool], bool]=False,
                **kwargs):
        """Compile distiller.
        
        Args:
            `optimizer_student`: Optimizer for student model(s).
                Defaults to `keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8)`.
            `optimizer_generator`: Optimizer for generator model.
                Defaults to `keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8)`.
            `onehot_loss_fn`:  One-hot loss function, as in original paper.
                Options:
                    `True`: use default `keras.losses.SparseCategoricalCrossentropy()`
                    `False`: toggle off
                    Others: custom user-defined loss function.
                Defaults to `True`.
            `activation_loss_fn`: Activation loss function, as in original paper.
                Options:
                    `True`: Use default (see `_activation_loss_fn`)
                    `False`: Toggle off
                    Others: Custom user-defined loss function
                Defaults to `True`.
            `info_entropy_loss_fn`: Information entropy loss function, as in original
            paper.
                Options:
                    `True`: Use default (see `_info_entropy_loss_fn`)
                    `False`: Toggle off
                    Others: Custom user-defined loss function
                Defaults to `True`.
            `distill_loss_fn`: Distillation loss function.
                Defaults to `keras.losses.KLDivergence()`.
            `student_loss_fn`: Loss function to evaluate the student's performance on
            the validation set.
                Defaults to `keras.losses.SparseCategoricalCrossentropy()`.
            `batch_size`: Size of each synthetic batch. Defaults to `512`.
            `num_batches`: Number of training batches each epoch. Defaults to `120`.
            `alpha`: Coefficient of activation loss. Defaults to `0.1`.
            `beta`: Coefficient of information entropy loss. Defaults to `5`.
            `temperature`: Temperature for label smoothing during distillation.
                Defaults to `1`.
            `confidence`: Confidence threshold for filtering out low-quality synthetic
            images (evaluated by the teacher) before distillation.
                Options:
                    `None`: do not apply
                    `float` number in the range [0, 1]: apply with one threshold
                    List of `None` and/or `float` numbers: apply all simultaneously
                Defaults to `None`.
            `rebalance`: Flag(s) to trigger rebalancing (dropping examples in a
            synthetic batch so every classes has the same number of examples with the
            least populated class).
                Options:
                    `False`: do not apply
                    `True`: do apply
                    `[False, True]`: apply both cases simultaneously
                Defaults to `False`.
        """
        if not isinstance(onehot_loss_fn, (keras.losses.Loss, bool)):
            warnings.warn('`onehot_loss_fn` should be of type `keras.losses.Loss` or `bool`.')
        if not isinstance(activation_loss_fn, (keras.losses.Loss, bool)):
            warnings.warn('`activation_loss_fn` should be of type `keras.losses.Loss` or `bool`.')
        if not isinstance(info_entropy_loss_fn, (keras.losses.Loss, bool)):
            warnings.warn('`info_entropy_loss_fn` should be of type `keras.losses.Loss` or `bool`.')

        super().compile(**kwargs)
        self.optimizer_student = optimizer_student
        self.optimizer_generator = optimizer_generator
        self.onehot_loss_fn = onehot_loss_fn
        self.activation_loss_fn = activation_loss_fn
        self.info_entropy_loss_fn = info_entropy_loss_fn        
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.alpha = alpha
        self.beta = beta
        # self.temperature = temperature

        # Handling multiple values of confidence and rebalance with multiple students & their optimizers:
        if isinstance(confidence, list):
            self.confidence = confidence
        elif not isinstance(confidence, list):
            self.confidence = [confidence]
        if isinstance(rebalance, list):
            self.rebalance = rebalance
        elif not isinstance(rebalance, list):
            self.rebalance = [rebalance]

        # Config one-hot loss
        if self.onehot_loss_fn is True:
            self._onehot_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        elif self.onehot_loss_fn is False:
            self._onehot_loss_fn = lambda *args, **kwargs:0
        else:
            self._onehot_loss_fn = self.onehot_loss_fn
        # Config activation loss
        if self.activation_loss_fn is True:
            pass
        elif self.activation_loss_fn is False:
            self._activation_loss_fn = lambda *args, **kwargs:0
        else:
            self._activation_loss_fn = self.activation_loss_fn
        # Config information entropy loss
        if self.info_entropy_loss_fn is True:
            pass
        elif self.info_entropy_loss_fn is False:
            self._info_entropy_loss_fn = lambda *args, **kwargs:0
        else:
            self._info_entropy_loss_fn = self.info_entropy_loss_fn

        # Placeholder data generator
        self.train_data = PlaceholderDataGenerator(
            num_batches=self.num_batches,
            batch_size=self.batch_size
        )

        # Clone students & optimizers with same parameters
        self.student.save('./logs/student_init')
        self.students:List[self.student.__class__] = []
        for confidence in self.confidence:
            for rebalance in self.rebalance:
                student:self.student.__class__ = keras.models.load_model('./logs/student_init')
                student.confidence = confidence
                student.rebalance = rebalance
                if student.rebalance is True:
                    student.name_suffix = f'{confidence}_R'
                elif student.rebalance is False:
                    student.name_suffix = f'{confidence}'

                self.students.append(student)
        
        optimizer_student_config = self.optimizer_student.get_config()
        self.optimizers_student = {
            student.name_suffix:self.optimizer_student.__class__(**optimizer_student_config) for student in self.students
        }

        # Metrics
        self.loss_onehot_metric = keras.metrics.Mean(name='loss_oh')
        self.loss_activation_metric = keras.metrics.Mean(name='loss_ac')
        self.loss_info_entropy_metric = keras.metrics.Mean(name='loss_ie')
        self.loss_generator_metric = keras.metrics.Mean(name='loss_gen')

        self.loss_distill_metric = {
            student.name_suffix:keras.metrics.Mean(name=f'loss_dt_{student.name_suffix}') for student in self.students
        }
        self.loss_ie_filtered_metric = {
            student.name_suffix:keras.metrics.Mean(name=f'loss_ie_filtered_{student.name_suffix}') for student in self.students
        }
        self.accuracy_metric = {
            student.name_suffix:keras.metrics.Accuracy(name=f'accuracy_{student.name_suffix}') for student in self.students
        }
        self.loss_student_metric = {
            student.name_suffix:keras.metrics.Mean(name=f'loss_student_{student.name_suffix}') for student in self.students
        }

    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring training step.
        
        Returns:
            List of training metrics.
        """
        return [self.loss_onehot_metric,
                self.loss_activation_metric,
                self.loss_info_entropy_metric,
                self.loss_generator_metric,
                *self.loss_ie_filtered_metric.values(),
                *self.loss_distill_metric.values()]
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring validation step.
        
        Returns:
            List of validation metrics.
        """
        return [*self.accuracy_metric.values(),
                *self.loss_student_metric.values()]

    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
            # Specify trainable variables
            tape.watch(self.generator.trainable_variables)
            for student in self.students:
                tape.watch(student.trainable_variables)

            # Phase 1 - Training the Generator
            latent_noise = tf.random.normal(shape=[self.batch_size, self.generator.latent_dim])
            x_synthetic = self.generator(latent_noise, training=True)
            teacher_prob, teacher_fmap = self.teacher(x_synthetic, training=False) # [512, 10], [512, 120]
            pseudo_label = tf.math.argmax(input=teacher_prob, axis=1)

            loss_onehot = self._onehot_loss_fn(pseudo_label, teacher_prob)
            loss_activation = self._activation_loss_fn(teacher_fmap)
            loss_info_entropy = self._info_entropy_loss_fn(teacher_prob)
            loss_generator = loss_onehot + self.alpha*loss_activation + self.beta*loss_info_entropy
            
            # Phase 2: Training the student network.
            loss_distill = {}
            loss_ie_filtered = {}
            for student in self.students:
                # Detach gradient graph of generator and teacher
                # Suffix "_f" for "filtered"
                x_synthetic_f  = tf.stop_gradient(tf.identity(x_synthetic))
                teacher_prob_f = tf.stop_gradient(tf.identity(teacher_prob))
                pseudo_label_f = tf.stop_gradient(tf.identity(pseudo_label))

                # Filter by confidence
                if student.confidence is not None:
                    confident_idx = tf.squeeze(
                        tf.where(tf.math.reduce_max(teacher_prob_f, axis=1) >= student.confidence),
                        axis=1
                    )
                    x_synthetic_f  = tf.gather(params=x_synthetic_f, indices=confident_idx)
                    teacher_prob_f = tf.gather(params=teacher_prob_f, indices=confident_idx)
                    pseudo_label_f = tf.gather(params=pseudo_label_f, indices=confident_idx)

                # Rebalancing to the least frequent class:
                if student.rebalance is True:
                    rebalance_idx = self.find_rebalance_idx(pred_tensor=pseudo_label_f, num_classes=10)
                    x_synthetic_f  = tf.gather(params=x_synthetic_f, indices=rebalance_idx)
                    teacher_prob_f = tf.gather(params=teacher_prob_f, indices=rebalance_idx)
                    pseudo_label_f = tf.gather(params=pseudo_label_f, indices=rebalance_idx)
                    
                student_prob = student(x_synthetic_f, training=True)
                # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
                # The magnitudes of the gradients produced by the soft targets scale
                # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
                loss_distill[student.name_suffix] = self.distill_loss_fn(teacher_prob_f, student_prob)
                loss_ie_filtered[student.name_suffix] = self._info_entropy_loss_fn(teacher_prob_f)
        
        ## Back-propagation of Generator
        generator_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_generator, generator_vars)
        self.optimizer_generator.apply_gradients(zip(gradients, generator_vars))
        # Back-propagation of Students
        for student in self.students:
            student_vars = student.trainable_variables
            gradients = tape.gradient(loss_distill[student.name_suffix], student_vars)
            self.optimizers_student[student.name_suffix].apply_gradients(zip(gradients, student_vars))
        del tape

        # Update the metrics, configured in 'compile()'
        self.loss_onehot_metric.update_state(loss_onehot)
        self.loss_activation_metric.update_state(loss_activation)
        self.loss_info_entropy_metric.update_state(loss_info_entropy)
        self.loss_generator_metric.update_state(loss_generator)
        for student in self.students:
            self.loss_distill_metric[student.name_suffix].update_state(loss_distill[student.name_suffix])
            self.loss_ie_filtered_metric[student.name_suffix].update_state(loss_ie_filtered[student.name_suffix])
        
        results = {m.name: m.result() for m in self.train_metrics}
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        for student in self.students:
            # Compute predictions and log accuracy
            student_prob = student(x, training=False)
            self.accuracy_metric[student.name_suffix].update_state(
                y_true=y,
                y_pred=tf.math.argmax(input=student_prob, axis=1)
            )
            # Compute and log loss
            self.loss_student_metric[student.name_suffix].update_state(
                values=self.student_loss_fn(y, student_prob)
            )
        
        results = {m.name: m.result() for m in self.val_metrics}
        return results

    def fit(self, **kwargs):
        super().fit(x=self.train_data, steps_per_epoch=self.num_batches, **kwargs)

    @staticmethod
    def find_rebalance_idx(pred_tensor:tf.Tensor, num_classes:int) -> tf.Tensor:        
        hist = tf.histogram_fixed_width(
            tf.cast(pred_tensor, dtype=tf.float32),
            value_range = [-0.5, num_classes-0.5], nbins=num_classes)
        num_least_class = tf.math.reduce_min(hist)

        shuffled_rebalanced_idx = [None for i in range(num_classes)]
        for label in range(num_classes):
            idx_by_label = tf.squeeze(tf.where(pred_tensor == label), axis=1)
            shuffled_rebalanced_idx[label] = idx_by_label[0:num_least_class]
        shuffled_rebalanced_idx = tf.concat(shuffled_rebalanced_idx, axis=0)

        return shuffled_rebalanced_idx

    @staticmethod
    def _activation_loss_fn(inputs:tf.Tensor) -> tf.Tensor:        
        """Activation loss function. Typical used with the teacher model's
        flattened feature map.
        
        Args:
            `inputs`: teacher model's flattened feature map. 
        Returns:
            Loss value.
        """        
        loss = -tf.math.reduce_mean(tf.abs(inputs), axis=[0, 1])
        return loss

    @staticmethod
    def _info_entropy_loss_fn(inputs):
        """Information entropy loss function. Typically used with the teacher model's
        prediction (probability).
        
        Args:
            `inputs`: teacher model's prediction (probability).
        Returns:
            Loss value.
        """        
        distr_synthetic = tf.math.reduce_mean(inputs, axis=0)
        loss = tf.math.reduce_sum(distr_synthetic*tf.experimental.numpy.log10(distr_synthetic))
        return loss

if __name__ == '__main__':
    from dataloader import dataloader
    from models.classifiers.LeNet_5 import LeNet_5_ReLU_MaxPool
    from models.classifiers.ResNet_DAFL import ResNet_DAFL
    from models.distillers.utils import CSVLogger_custom, add_fmap_output
    from models.GANs.utils import MakeSyntheticGIFCallback

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
        LATENT_DIM = 100
        IMAGE_DIM = [32, 32, 1] # LeNet-5 accepts [32, 32] images
        NUM_CLASSES = 10
        BATCH_SIZE_TEACHER, BATCH_SIZE_DISTILL = 256, 512
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 10, 200
        OPTIMIZER_TEACHER = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
        OPTIMIZER_GENERATOR = keras.optimizers.Adam(learning_rate=2e-1, epsilon=1e-8)
        OPTIMIZER_STUDENT = keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8)
        COEFF_OH, COEFF_AC, COEFF_IE = 1, 0.1, 5
        
        print(' Experiment 4.1: DAFL on MNIST. Teacher: LeNet-5, student: LeNet-5-HALF '.center(80,'#'))
        
        ds = dataloader(
            dataset='mnist',
            resize=IMAGE_DIM[0:-1],
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
            batch_size_test=1024
        )

        # Teacher (LeNet-5)
        teacher = LeNet_5_ReLU_MaxPool(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        teacher.compile(
            metrics=['accuracy'], 
            optimizer=OPTIMIZER_TEACHER,
            loss=keras.losses.SparseCategoricalCrossentropy())
        teacher.build()

        if pretrained_teacher is True:
            teacher.load_weights('./pretrained/mnist/mean0_std1/LeNet-5_ReLU_MaxPool_9900.h5')
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
                epochs=NUM_EPOCHS_TEACHER,
                callbacks=[best_callback, csv_logger],
                shuffle=True,
                validation_data=ds['test']
            )
            teacher.load_weights(filepath=f'./logs/{teacher.name}_best.h5')
        teacher.evaluate(ds['test'])
        teacher = add_fmap_output(model=teacher, fmap_layer='flatten')

        # Student (LeNet-5-HALF)
        student = LeNet_5_ReLU_MaxPool(half=True, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        student.build()
        student.compile(metrics='accuracy')
        student.evaluate(ds['test'])

        generator = DataFreeGenerator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)
        generator.build()

        # Train one student with default data-free learning settings
        distiller = DataFreeDistiller(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=OPTIMIZER_STUDENT,
            optimizer_generator=OPTIMIZER_GENERATOR,
            onehot_loss_fn=True,
            activation_loss_fn=True,
            info_entropy_loss_fn=True,
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_oh=COEFF_OH,
            coeff_ac=COEFF_AC,
            coeff_ie=COEFF_IE,
            confidence=None
        )

        csv_logger = CSVLogger_custom(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        gif_maker = MakeSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.gif',
            nrows=5, ncols=5,
            postprocess_fn=lambda x:x*0.3081 + 0.1307,
            normalize=False,
            save_freq=NUM_EPOCHS_DISTILL//50
        )

        distiller.fit(
            epochs=NUM_EPOCHS_DISTILL,
            callbacks=[csv_logger, gif_maker],
            shuffle=True,
            validation_data=ds['test']
        )

    def run_experiment_cifar10(pretrained_teacher:bool=False):
        # Algorithm                     Required data   FLOPS   params  CIFAR-10  CIFAR-100
        # Teacher                       Original data   16G     21M     95.58%    77.84%
        # Standard back-propagation     Original data   557M    11M     93.92%    76.53%
        # Knowledge Distillation [8]    Original data   557M    11M     94.34%    76.87%
        # Normal distribution           No data         557M    11M     14.89%     1.44%
        # Alternative data              Similar data    557M    11M     90.65%    69.88%
        # Data-Free Learning (DAFL)     No data         557M    11M     92.22%    74.47%
        LATENT_DIM = 1000
        IMAGE_DIM = [32, 32, 3]
        NUM_CLASSES = 10
        BATCH_SIZE_TEACHER, BATCH_SIZE_DISTILL = 128, 1024
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 200, 2000
        COEFF_OH, COEFF_AC, COEFF_IE = 0.05, 0.01, 5

        OPTIMIZER_TEACHER = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9) # 1e-1 to 1e-2 to 1e-3
        OPTIMIZER_GENERATOR = keras.optimizers.Adam(learning_rate=2e-2, epsilon=1e-8)
        OPTIMIZER_STUDENT = keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9)

        print(' Experiment 4.4: DAFL on CIFAR-10. Teacher: ResNet-34, student: ResNet-18 '.center(80,'#'))

        def augmentation_fn(x):
            x = tf.pad(tensor=x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
            x = tf.image.random_crop(value=x, size=[tf.shape(x)[0], 32, 32, 3])
            x = tf.image.random_flip_left_right(image=x)
            return x
        ds = dataloader(
            dataset='cifar10',
            augmentation_fn=augmentation_fn,
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
            batch_size_test=1024
        )

        # Teacher (ResNet-34)
        teacher = ResNet_DAFL(ver=34, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        teacher.build()
        teacher.compile(
                metrics=['accuracy'], 
                optimizer=OPTIMIZER_TEACHER,
                loss=keras.losses.SparseCategoricalCrossentropy())

        if pretrained_teacher is True:
            teacher.load_weights(filepath=f'./pretrained/cifar10/mean0_std1/ResNet-DAFL-34_9499.h5')
        elif pretrained_teacher is False:
            def schedule(epoch:int, learing_rate:float):
                if epoch in [80, 120]:
                    learing_rate = learing_rate*0.1
                return learing_rate
            lr_scheduler = LearningRateSchedulerCustom(
                schedule=schedule,
                optimizer_name='optimizer',
                verbose=1)
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
                epochs=NUM_EPOCHS_TEACHER,
                callbacks=[best_callback, lr_scheduler, csv_logger],
                validation_data=ds['test']
            )
            teacher.load_weights(filepath=f'./logs/{teacher.name}_best.h5')
        teacher.evaluate(ds['test'])
        teacher = add_fmap_output(model=teacher, fmap_layer='glb_pool')

        # Student (ResNet-18)
        student = ResNet_DAFL(ver=18, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        student.build()
        student.compile(metrics='accuracy')
        student.evaluate(ds['test'])

        generator = DataFreeGenerator(latent_dim=LATENT_DIM, image_dim=IMAGE_DIM)
        generator.build()

        # Train one student with default data-free learning settings
        distiller = DataFreeDistiller(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=OPTIMIZER_STUDENT,
            optimizer_generator=OPTIMIZER_GENERATOR,
            onehot_loss_fn=True,
            activation_loss_fn=True,
            info_entropy_loss_fn=True,
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_oh=COEFF_OH,
            coeff_ac=COEFF_AC,
            coeff_ie=COEFF_IE,
            confidence=None
        )

        def schedule(epoch:int, learing_rate:float):
            if epoch in [800, 1600]:
                learing_rate = learing_rate*0.1
            return learing_rate
        lr_scheduler = LearningRateSchedulerCustom(
            schedule=schedule,
            optimizer_name='optimizer_student',
            verbose=1)
        csv_logger = CSVLogger_custom(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        gif_maker = MakeSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.gif',
            nrows=5, ncols=5,
            postprocess_fn=lambda x:x*tf.constant([[[0.2470, 0.2435, 0.2616]]]) + tf.constant([[[0.4914, 0.4822, 0.4465]]]),
            normalize=False,
            # save_freq=NUM_EPOCHS_DISTILL//50
        )

        distiller.fit(
            epochs=NUM_EPOCHS_DISTILL,
            callbacks=[lr_scheduler, csv_logger, gif_maker],
            shuffle=True,
            validation_data=ds['test']
        )

    run_experiment_cifar10(pretrained_teacher=True)