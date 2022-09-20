from typing import List
import tensorflow as tf

keras = tf.keras

# import numpy as np
# rng = np.random.default_rng(seed=17)
# RAND_FEATURES = rng.normal(loc=0, scale=1, size=[512, 8, 8, 128]).astype('float32')
# RAND_FEATURES = tf.convert_to_tensor(RAND_FEATURES)
# print(RAND_FEATURES[0, 0, 0, -3:])
# RAND_OUTPUTS = rng.normal(loc=0, scale=1, size=[512, 10]).astype('float32')
# RAND_FEATURES = rng.normal(loc=0, scale=1, size=[512, 120]).astype('float32')
# print(RAND_OUTPUTS[0, 0], RAND_OUTPUTS[-1, -1], RAND_FEATURES[0, 0], RAND_FEATURES[-1, -1])

class DataFreeGenerator(keras.Model):
    """DCGAN Generator model implemented in Data-Free Learning of Student
    Networks - Chen et al. (2019), replicated with the same architecture.

    Original: Unsupervised representation learning with deep convolutional
    generative adversarial networks - Radford et al. (2015)
    DOI: 10.48550/arXiv.1511.06434
    """    
    _name = 'DataFreeGenerator'

    def __init__(self,
                 latent_dim:List[int]=[100],
                 image_dim:List[int]=[32, 32, 3],

                 *args, **kwargs):
        """Initialize generator.

        Args:
            latent_dim (List[int], optional): Dimension of latent space. Defaults to [100].
            image_dim (List[int], optional): Dimension of output images. Defaults to [32, 32, 3].
        """
        super().__init__(self, name=self._name, *args, **kwargs)
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.init_dim = [self.image_dim[0]//4, self.image_dim[1]//4]

        self.dense = keras.layers.Dense(units=self.init_dim[0] * self.init_dim[1] * 128)
        self.reshape = keras.layers.Reshape(target_shape=(self.init_dim[0], self.init_dim[1], 128))
        self.conv_block_0 = keras.Sequential(
            layers=[keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-05)],
            name='conv_block_0'
        )

        self.interpl_1 = keras.layers.Resizing(
            height=2*self.init_dim[0],
            width=2*self.init_dim[1],
            interpolation='nearest',
            crop_to_aspect_ratio=False)
        self.conv_block_1 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
                keras.layers.BatchNormalization(momentum=0.9, epsilon=0.8),
                keras.layers.LeakyReLU(alpha=0.2)
            ],
            name='conv_block_1'
        )
        
        self.interpl_2    = keras.layers.Resizing(
            height=4*self.init_dim[0],
            width=4*self.init_dim[1],
            interpolation='nearest',
            crop_to_aspect_ratio=False)
        self.conv_block_2 = keras.Sequential(
            layers=[
                keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'),
                keras.layers.BatchNormalization(momentum=0.9, epsilon=0.8),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(filters=self.image_dim[2], kernel_size=3, strides=1, padding='same'),
                keras.layers.Activation(tf.nn.tanh),
                keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)],
            name='conv_block_2'
        )
        
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_block_0(x)
        x = self.interpl_1(x)
        x = self.conv_block_1(x)
        x = self.interpl_2(x)
        x = self.conv_block_2(x)
        # x = (x + 1)/2     # Authors do not normalize to [0, 1)
        return x

    def build(self):
        super().build(input_shape=[None]+self.latent_dim)
        inputs = keras.layers.Input(shape=self.latent_dim)
        self.call(inputs)

class DataFreeDistiller(keras.Model):
    """Data-Free Learning of Student Networks - Chen et al. (2019)
    DOI: 10.48550/arXiv.1904.01186

    A knowledge distillation (KD) scheme performed without the training set
    and architecture information of the teacher model, utilizing a generator
    approximating the original dataset.

    Implementation in PyTorch: https://github.com/autogyro/DAFL
    """
    _name = 'DataFreeDistiller'

    def __init__(self,
                 teacher:keras.Model,
                 student:keras.Model,
                 generator:DataFreeGenerator,
                 *args, **kwargs):
        """Initialize distiller.

        Args:
            teacher (keras.Model): Pre-trained teacher model
            student (keras.Model): To-be-trained student model
            generator (DataFreeGenerator): DCGAN generator proposed in study
        """                 
        super().__init__(name=self._name, *args, **kwargs)
        self.teacher = teacher
        self.student = student
        self.generator = generator

    def compile(self,
                batch_size:int,
                optimizer_student:keras.optimizers.Optimizer, # = keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8),
                optimizer_generator:keras.optimizers.Optimizer, # = keras.optimizers.Adam(learning_rate=2e-1, epsilon=1e-8),
                onehot_loss_fn:keras.losses.Loss,
                activation_loss_fn:keras.losses.Loss, # L1-norm
                info_entropy_loss_fn:keras.losses.Loss, # information entropy loss
                distillation_loss_fn:keras.losses.Loss,
                student_loss_fn:keras.losses.Loss,
                alpha:float=0.1,
                beta:float=5,
                temperature:float=1,
                confidence:float=None):
        """Compile distiller.

        Args:
            batch_size (int): Batch size of synthetic images.
            optimizer_student (keras.optimizers.Optimizer): Optimizer for student model.
            optimizer_generator (keras.optimizers.Optimizer): Optimizer for generator model.
            onehot_loss_fn (keras.losses.Loss): Loss function for one-hot loss, helps
                generator produce distinctive images.
            activation_loss_fn (keras.losses.Loss): Loss function for activation loss, helps
                guide generator towards realistic synthesis of images.
            info_entropy_loss_fn (keras.losses.Loss): Loss function for information entropy
                loss, helps generator produce same amount of images for each class.
            distillation_loss_fn (keras.losses.Loss): Loss function for distillation, helps
                student match the teacher's prediction
            student_loss_fn (keras.losses.Loss): Loss function for evaluating the student's
                performance on the test set.
            alpha (float, optional): Coefficient of activation loss. Defaults to 0.1.
            beta (float, optional): Coefficient of information entropy loss. Defaults to 5.
            temperature (float, optional): Temperature for label smoothing during
                distillation. Defaults to 10.
            confidence (float, optional): Confidence threshold for filtering out low-quality
                synthetic images (evaluated by the teacher) before distillation. Defaults to
                None.
        """        
        super().compile()
        self.batch_size = batch_size
        self.optimizer_student = optimizer_student
        self.optimizer_generator = optimizer_generator
        self.onehot_loss_fn = onehot_loss_fn
        self.activation_loss_fn = activation_loss_fn
        self.info_entropy_loss_fn = info_entropy_loss_fn        
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.confidence = confidence

        # Metrics
        self.loss_onehot_metric = keras.metrics.Mean(name='loss_onehot')
        self.loss_activation_metric = keras.metrics.Mean(name='loss_activation')
        self.loss_info_entropy_metric = keras.metrics.Mean(name='loss_info_entropy')
        self.loss_generator_metric = keras.metrics.Mean(name='loss_generator')
        self.loss_distill_metric = keras.metrics.Mean(name='loss_distill')

        self.accuracy_metric = keras.metrics.Accuracy(name='accuracy')
        self.loss_student_metric = keras.metrics.Mean(name='loss_student')

    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring training step.

        Returns:
            List[keras.metrics.Metric]: List of training metrics
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
            List[keras.metrics.Metric]: List of validation metrics
        """        
        return [self.accuracy_metric,
                self.loss_student_metric]

    def train_step(self, data):
        batch_size = self.batch_size

        with tf.GradientTape(persistent=True) as tape:
            # Phase 1 - Training the Generator
            '''
            Randomly generate a batch of vector:
            Generate the training samples: x ← G(z);
            Employ the teacher network on the mini-batch: [yT , t, fT ] ← NT (x);
            Calculate the loss function LT otal (Fcn.7):
            Update weights in G using back-propagation;
            '''
            latent_noise = tf.random.normal(shape=[batch_size, self.generator.latent_dim[0]])
            x_synthetic = self.generator(latent_noise, training=True)
            teacher_logits, teacher_fmap = self.teacher(x_synthetic, training=False) # [512, 10], [512, 120]

            # teacher_logits, teacher_fmap = tf.convert_to_tensor(RAND_OUTPUTS), tf.convert_to_tensor(RAND_FEATURES)

            # Teacher prediction is used as pseudo-label
            teacher_pred = tf.math.argmax(input=teacher_logits, axis=1)

            loss_onehot = self.onehot_loss_fn(teacher_pred, teacher_logits)
            loss_activation = -tf.math.reduce_mean(tf.norm(teacher_fmap, ord=1, axis=1)/120, axis=0)
            
            teacher_prob = tf.nn.softmax(logits=teacher_logits, axis=1)
            distr_synthetic = tf.math.reduce_mean(teacher_prob, axis=0)
            loss_info_entropy = tf.math.reduce_sum(distr_synthetic*tf.experimental.numpy.log10(distr_synthetic))
            
            loss_generator = loss_onehot + self.alpha*loss_activation + self.beta*loss_info_entropy
            
            # Phase 2: Training the student network.
            '''
            Randomly generate a batch of vector {zi}n i=1
            Utlize the generator on the mini-batch: x ← G(z)
            Employ the teacher network and the student network on the mini-batch simultaneously
            yS ← NS(x), yT ← NT (x)
            Calculate the knowledge distillation loss
            Update weights in NS according to the gradient
            '''
            # Detach gradient graph of generator and teacher
            x_synthetic = tf.stop_gradient(tf.identity(x_synthetic))
            teacher_logits = tf.stop_gradient(tf.identity(teacher_logits))

            if self.confidence is not None:
                # Keep only images with high confidence to train student
                teacher_prob = tf.stop_gradient(tf.identity(teacher_prob))
                confident_idx = tf.squeeze(tf.where(tf.math.reduce_max(teacher_prob, axis=1) > self.confidence), axis=1)

                x_synthetic = tf.gather(params=x_synthetic, indices=confident_idx)
                teacher_logits = tf.gather(params=teacher_logits, indices=confident_idx)

            student_logits = self.student(x_synthetic, training=True)
            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            loss_distill = self.temperature**2 * self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1))
        
        ## Back-propagation of Generator
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

    def train_step_exact(self, data):
        batch_size = self.batch_size

        with tf.GradientTape() as tape:
            # Phase 1 - Training the Generator
            '''
            Randomly generate a batch of vector:
            Generate the training samples: x ← G(z);
            Employ the teacher network on the mini-batch: [yT , t, fT ] ← NT (x);
            Calculate the loss function LT otal (Fcn.7):
            Update weights in G using back-propagation;
            '''
            latent_noise = tf.random.normal(shape=[batch_size, self.generator.latent_dim[0]])
            x_synthetic = self.generator(latent_noise, training=True)
            teacher_logits, teacher_fmap = self.teacher(x_synthetic, training=False) # [512, 10], [512, 120]

            # Teacher prediction is used as pseudo-label
            teacher_pred = tf.math.argmax(input=teacher_logits, axis=1)

            loss_onehot = self.onehot_loss_fn(teacher_pred, teacher_logits)
            
            loss_activation = -tf.math.reduce_mean(tf.norm(teacher_fmap, ord=1, axis=1)/120, axis=0)
            
            teacher_prob = tf.nn.softmax(logits=teacher_logits, axis=1)
            distr_synthetic = tf.math.reduce_mean(teacher_prob, axis=0)
            loss_info_entropy = tf.math.reduce_sum(distr_synthetic*tf.experimental.numpy.log10(distr_synthetic))
            
            loss_generator = loss_onehot + self.alpha*loss_activation + self.beta*loss_info_entropy
        # Back-propagation of Generator
        generator_vars = self.generator.trainable_variables
        gradients = tape.gradient(loss_generator, generator_vars)
        self.optimizer_generator.apply_gradients(zip(gradients, generator_vars))
        
        
        # Phase 2: Training the student network.
        '''
        Randomly generate a batch of vector {zi}n i=1
        Utlize the generator on the mini-batch: x ← G(z)
        Employ the teacher network and the student network on the mini-batch simultaneously
        yS ← NS(x), yT ← NT (x)
        Calculate the knowledge distillation loss
        Update weights in NS according to the gradient
        '''
        latent_noise = tf.random.normal(shape=[batch_size, self.generator.latent_dim[0]])
        x_synthetic = self.generator(latent_noise, training=False)
        teacher_logits, _ = self.teacher(x_synthetic, training=False) # [512, 10], [512, 120]
        with tf.GradientTape() as tape:
            student_logits = self.student(x_synthetic, training=True)
            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            loss_distill = self.temperature**2 * self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1))
        # Back-propagation of Student
        student_vars = self.student.trainable_variables
        gradients = tape.gradient(loss_distill, student_vars)
        self.optimizer_student.apply_gradients(zip(gradients, student_vars))

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
        student_logits = self.student(x, training=False)
        student_pred = tf.math.argmax(input=student_logits, axis=1)

        # Calculate the loss
        loss_student = self.student_loss_fn(y, student_logits)
        
        # Update the metrics, configured in 'compile()'
        self.accuracy_metric.update_state(y_true=y, y_pred=student_pred)
        self.loss_student_metric.update_state(loss_student)
        results = {m.name: m.result() for m in self.val_metrics}
        return results

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    import tensorflow_datasets as tfds
    from models.LeNet_5 import LeNet_5, LeNet_5_ReLU_MaxPool

    tf.config.run_functions_eagerly(True)

    # Hyperparameters
    ## Model
    LATENT_DIM = [100]
    IMAGE_DIM = [32, 32, 1]
    ALPHA = 0.1
    BETA = 5
    TEMPERATURE = 1
    ## Training
    LEARNING_RATE_STUDENT = 2e-3
    LEARNING_RATE_GENERATOR = 0.2
    BATCH_SIZE = 512
    NUM_EPOCHS = 200

    def add_fmap_output(model:keras.Model, fmap_layer:str) -> keras.Model:
        """Extract a model's feature map and add to its output.

        Args:
            model (keras.Model): host model of feature map
            fmap_layer (str): name of feature map layer

        Returns:
            keras.Model: host model with an additional feature map output
        """        
        model.build()
        model = keras.Model(
            inputs=model.layers[0].input,
            outputs=[model.layers[-1].output, model.get_layer(fmap_layer).output],
            name=model.name)
        return model

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

    #@title CSVLogger
    import numpy as np
    from keras.utils import io_utils
    import collections
    import csv
    class CSVLogger(keras.callbacks.CSVLogger):
        """Re-implementation of keras.callbacks.CSVLogger to use with model.fit() when
        validation_freq > 1.
        
        Args:
            filename: Filename of the CSV file, e.g. `'run/log.csv'`.
            separator: String used to separate elements in the CSV file.
            append: Boolean. True: append if file exists (useful for continuing
                training). False: overwrite existing file.
        """

        def __init__(self, filename, separator=",", append=False):
            self.sep = separator
            self.filename = io_utils.path_to_string(filename)
            self.append = append
            self.writer = None
            self.keys = None
            self.append_header = True
            super().__init__()

        def on_train_begin(self, logs=None):
            if self.append:
                if tf.io.gfile.exists(self.filename):
                    with tf.io.gfile.GFile(self.filename, "r") as f:
                        self.append_header = not bool(len(f.readline()))
                mode = "a"
            else:
                mode = "w"
            self.csv_file = tf.io.gfile.GFile(self.filename, mode)

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, str):
                    return k
                elif (
                    isinstance(k, collections.abc.Iterable)
                    and not is_zero_dim_ndarray
                ):
                    return '"[%s]"' % (", ".join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict(
                    (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
                )

            if not self.writer:

                class CustomDialect(csv.excel):
                    delimiter = self.sep

                fieldnames = ["epoch"] + self.keys

                self.writer = csv.DictWriter(
                    self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
                )
                if self.append_header:
                    self.writer.writeheader()

            row_dict = collections.OrderedDict({"epoch": epoch})
            row_dict.update((key, handle_value(logs[key])) if key in logs else (key, "") for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        def on_train_end(self, logs=None):
            self.csv_file.close()
            self.writer = None

    ds = tfds.load('mnist', as_supervised=True)
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)/255.
        x = tf.image.resize(x, IMAGE_DIM[0:2])
        x = (x - 0.1307)/0.3081
        return x, y
    # ds['train'] = ds['train'].map(preprocess).shuffle(60000).batch(BATCH_SIZE).prefetch(1)
    ds['test'] = ds['test'].map(preprocess).batch(BATCH_SIZE).prefetch(1)

    # Pretrained teacher (LeNet-5 traditional) 99.06%
    teacher = LeNet_5()
    teacher.build()
    teacher.load_weights('./pretrained/MNIST/LeNet-5_tanh_AvgPool_9906.h5')
    teacher.compile(metrics='accuracy')
    teacher.evaluate(ds['test'])
    teacher = add_fmap_output(model=teacher, fmap_layer='flatten')

    # Student (LeNet-5-HALF)
    student = LeNet_5(half=True)
    student.build()
    student.compile(metrics='accuracy')
    student.evaluate(ds['test'])

    generator = DataFreeGenerator(latent_dim=[100], image_dim=[32, 32, 1])
    generator.build()

    distiller = DataFreeDistiller(
        teacher=teacher,
        student=student,
        generator=generator)
    distiller.compile(
        batch_size=BATCH_SIZE,
        optimizer_student=keras.optimizers.Adam(learning_rate=2e-3, epsilon=1e-8),
        optimizer_generator=keras.optimizers.Adam(learning_rate=0.2, epsilon=1e-8),
        onehot_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        activation_loss_fn=tf.norm,
        info_entropy_loss_fn=tf.math.log,
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha=ALPHA,
        beta=BETA,
        temperature=1,
        confidence=0.8)

    best_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./logs/{distiller.name}_best.ckpt',
        monitor='val_accuracy',
        verbose=1,
        # initial_value_threshold=0.9,
        save_best_only=True,
        save_weights_only=True,
    )
    csv_logger = CSVLogger(
        f'./logs/{distiller.name}_student.csv',
        append=True)
    distiller.fit(
        x=tf.random.normal(shape=(1, 1, 1, 1)),
        y=tf.random.normal(shape=(1, 1, 1, 1)),
        epochs=1,
        callbacks=[best_callback, csv_logger],
        verbose=1,
        shuffle=True,
        validation_data=ds['test'])
    distiller.fit(
        x=tf.random.normal(shape=(1, 1, 1, 1)),
        y=tf.random.normal(shape=(1, 1, 1, 1)),
        epochs=NUM_EPOCHS*120,
        initial_epoch=1,
        callbacks=[best_callback, csv_logger],
        verbose=1,
        shuffle=True,
        validation_data=ds['test'],
        validation_freq=120)