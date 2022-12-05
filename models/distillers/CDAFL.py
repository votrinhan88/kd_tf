import warnings
from typing import List, Union, Literal
import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.distillers.DataFreeDistiller import DataFreeDistiller, DataFreeGenerator
    from models.classifiers.ResNet_DAFL import ResNet_DAFL
else:
    from .DataFreeDistiller import DataFreeDistiller, DataFreeGenerator
    from ..classifiers.ResNet_DAFL import ResNet_DAFL

class ConditionalDataFreeGenerator(DataFreeGenerator):
    """Review GAN to CGAN to modify.
    """
    _name = 'CDAFL_Gen'
    def __init__(self,
                 latent_dim:int=100,
                 image_dim:List[int]=[32, 32, 3],
                 embed_dim:Union[int, None]=None,
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
        if self.embed_dim is None:
            self.label_branch = keras.Sequential([
                keras.layers.Dense(units=tf.math.reduce_prod(self._BASE_DIM)),
                keras.layers.Reshape(target_shape=(*self._BASE_DIM, 1)),
                keras.layers.ReLU(),
            ])
        elif isinstance(self.embed_dim, int):
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
                keras.layers.BatchNormalization(momentum=self._MOMENTUM, epsilon=self._EPSILONS[3], center=False, scale=False)
            ],
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
            keras.Model.summary(self, **kwargs)

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

class ConditionalLenet5_ReLU_MaxPool(keras.Model):
    _name = 'cLeNet5'
    def __init__(self,
                 half:bool=False,
                 input_dim:List[int]=[32, 32, 1],
                 embed_dim:Union[int, None]=None,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=False,
                 **kwargs):
        assert isinstance(half, bool), '`half` should be of type bool'
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'
        
        if half is False:
            keras.Model.__init__(self, name=self._name, **kwargs)
            divisor = 1
        elif half is True:
            keras.Model.__init__(self, name='LeNet-5-HALF_ReLU_MaxPool', **kwargs)
            divisor = 2
        
        self.half = half
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.onehot_input = onehot_input
        self.return_logits = return_logits

        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        if self.embed_dim is None:
            self.label_branch = keras.Sequential([
                keras.layers.Dense(units=tf.math.reduce_prod(self.input_dim[0:-1])),
                keras.layers.Reshape(target_shape=(*self.input_dim[0:-1], 1)),
                keras.layers.ReLU(),
            ])
        elif isinstance(self.embed_dim, int):
            self.label_branch = keras.Sequential([
                keras.layers.Dense(units=self.embed_dim),
                keras.layers.ReLU(),
                keras.layers.Dense(units=tf.math.reduce_prod(self.input_dim[0:-1])),
                keras.layers.Reshape(target_shape=(*self.input_dim[0:-1], 1)),
                keras.layers.ReLU(),
            ])
        self.concat = keras.layers.Concatenate()

        self.C1      = keras.layers.Conv2D(filters=6//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C1')
        self.S2      = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name='S2')
        self.C3      = keras.layers.Conv2D(filters=16//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C3')
        self.S4      = keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='valid', name='S4')
        self.C5      = keras.layers.Conv2D(filters=120//divisor, kernel_size=5, strides=1, activation='ReLU', padding='valid', name='C5')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.F6      = keras.layers.Dense(units=84//divisor, activation='ReLU', name='F6')
        self.pred    = keras.layers.Dense(units=1, name='pred', activation=tf.nn.sigmoid)

    def call(self, inputs, training:bool=False):
        images, labels = inputs

        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        label_branch = self.label_branch(labels)
        
        x = self.concat([images, label_branch])
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.pred(x)
        return x

    def build(self):
        if self.onehot_input is True:
            super().build(input_shape=[[None, *self.input_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            super().build(input_shape=[[None, *self.input_dim], [None, 1]])

    def summary(self, with_graph:bool=False, **kwargs):
        image_inputs = keras.layers.Input(shape=self.input_dim)
        if self.onehot_input is True:
            label_inputs = keras.layers.Input(shape=[self.num_classes])
        elif self.onehot_input is False:
            label_inputs = keras.layers.Input(shape=[1])
        inputs = [image_inputs, label_inputs]
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

class ConditionalResNet_DAFL(ResNet_DAFL):
    _name = 'cResNet-DAFL'
    def __init__(self,
                 ver:Literal[18, 34, 50, 101, 152]=18,
                 input_dim:List[int]=[32, 32, 3],
                 embed_dim:Union[int, None]=None,
                 num_classes:int=10,
                 onehot_input:bool=True,
                 return_logits:bool=False,
                 **kwargs):
        super().__init__(
            ver=ver,
            input_dim=input_dim,
            num_classes=1,  # Pass 1 in to receive one output node
            return_logits=return_logits,
            **kwargs)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.onehot_input = onehot_input

        if self.onehot_input is False:
            self.cate_encode = keras.layers.CategoryEncoding(num_tokens=self.num_classes, output_mode='one_hot')
        if self.embed_dim is None:
            self.label_branch = keras.Sequential([
                keras.layers.Dense(units=tf.math.reduce_prod(self.input_dim[0:-1])),
                keras.layers.Reshape(target_shape=(*self.input_dim[0:-1], 1)),
                keras.layers.ReLU(),
            ])
        elif isinstance(self.embed_dim, int):
            self.label_branch = keras.Sequential([
                keras.layers.Dense(units=self.embed_dim),
                keras.layers.ReLU(),
                keras.layers.Dense(units=tf.math.reduce_prod(self.input_dim[0:-1])),
                keras.layers.Reshape(target_shape=(*self.input_dim[0:-1], 1)),
                keras.layers.ReLU(),
            ])
        
        self.concat = keras.layers.Concatenate()

    def call(self, inputs, training:bool = False):
        images, labels = inputs

        if self.onehot_input is False:
            labels = self.cate_encode(labels)
        label_branch = self.label_branch(labels)

        x = self.concat([images, label_branch])
        x = super().call(x, training)
        return x

    def build(self):
        if self.onehot_input is True:
            keras.Model.build(self, input_shape=[[None, *self.input_dim], [None, self.num_classes]])
        elif self.onehot_input is False:
            keras.Model.build(self, input_shape=[[None, *self.input_dim], [None, 1]])

    def summary(self, with_graph:bool=False, **kwargs):
        image_inputs = keras.layers.Input(shape=self.input_dim)
        if self.onehot_input is True:
            label_inputs = keras.layers.Input(shape=[self.num_classes])
        elif self.onehot_input is False:
            label_inputs = keras.layers.Input(shape=[1])
        inputs = [image_inputs, label_inputs]
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            keras.Model.summary(self, **kwargs)

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
    _name = 'CDAFL'
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
        super().__init__(
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
                conditional_loss_fn:Union[bool, keras.losses.Loss]=True,
                distill_loss_fn:keras.losses.Loss=keras.losses.KLDivergence(),
                student_loss_fn:keras.losses.Loss=keras.losses.SparseCategoricalCrossentropy(),
                batch_size:int=500,
                num_batches:int=120,
                coeff_oh:float=1,
                coeff_ac:float=0.1,
                coeff_ie:float=5,
                coeff_cn:float=1,
                confidence:float=None,
                **kwargs):

        if batch_size % self.num_classes > 0:
            warnings.warn(
                f'`batch_size` {batch_size} is not divisible by `num_classes` '+
                f'{self.num_classes} and will give unevenly distributed batches.')

        super().compile(
            optimizer_student=optimizer_student,
            optimizer_generator=optimizer_generator,
            onehot_loss_fn=onehot_loss_fn,
            activation_loss_fn=activation_loss_fn,
            info_entropy_loss_fn=info_entropy_loss_fn,
            distill_loss_fn=distill_loss_fn,
            student_loss_fn=student_loss_fn,
            batch_size=batch_size,
            num_batches=num_batches,
            coeff_oh=coeff_oh,
            coeff_ac=coeff_ac,
            coeff_ie=coeff_ie,
            confidence=confidence,
            **kwargs)
        self.conditional_loss_fn = conditional_loss_fn
        self.coeff_cn = coeff_cn

        # Config conditional loss
        if self.conditional_loss_fn is True:
            if self.onehot_input is True:
                self._conditional_loss_fn = keras.losses.CategoricalCrossentropy()
            elif self.onehot_input is False:
                self._conditional_loss_fn = keras.losses.SparseCategoricalCrossentropy()
        elif self.conditional_loss_fn is False:
            self._conditional_loss_fn = lambda *args, **kwargs:0
        else:
            self._conditional_loss_fn = self.conditional_loss_fn

        # Additional metrics
        if self.conditional_loss_fn is not False:
            self.loss_conditional_metric = keras.metrics.Mean(name='loss_cn')
        
    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:        
        """Metrics monitoring training step.
        
        Returns:
            List of training metrics.
        """
        train_metrics = super().train_metrics
        if self.conditional_loss_fn is not False:
            train_metrics.append(self.loss_conditional_metric)
        return train_metrics

    def train_step(self, data):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            # Specify trainable variables
            tape.watch(self.generator.trainable_variables)
            tape.watch(self.student.trainable_variables)

            # Phase 1 - Training the conditional Generator
            latent_noise = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            # label = tf.cast(
            #     tf.math.floor(tf.linspace(start=0, stop=self.num_classes, num=self.batch_size+1)[0:-1]),
            #     dtype=tf.int32
            # )
            if self.onehot_input is True:
                # # Convert to one-hot label
                # oh_label = tf.one_hot(indices=label, depth=self.num_classes)
                label = tf.random.uniform(shape=[self.batch_size, self.num_classes])
                label = label/tf.tile(tf.math.reduce_sum(label, axis=1, keepdims=True), multiples=[1, self.num_classes])
                x_synth = self.generator([latent_noise, label], training=True)
            elif self.onehot_input is False:
                x_synth = self.generator([latent_noise, label], training=True)
            
            teacher_prob, teacher_fmap = self.teacher(x_synth, training=False)
            pseudo_label = tf.math.argmax(input=teacher_prob, axis=1)

            loss_onehot = self._onehot_loss_fn(pseudo_label, teacher_prob)
            loss_activation = self._activation_loss_fn(teacher_fmap)
            loss_info_entropy = self._info_entropy_loss_fn(teacher_prob)
            loss_conditional = self._conditional_loss_fn(label, teacher_prob)

            loss_generator = (
                self.coeff_oh*loss_onehot + 
                self.coeff_ac*loss_activation + 
                self.coeff_ie*loss_info_entropy +
                self.coeff_cn*loss_conditional
            )
            
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
        if self.onehot_loss_fn is not False:
            self.loss_onehot_metric.update_state(loss_onehot)
        if self.activation_loss_fn is not False:
            self.loss_activation_metric.update_state(loss_activation)
        if self.info_entropy_loss_fn is not False:
            self.loss_info_entropy_metric.update_state(loss_info_entropy)
        if self.conditional_loss_fn is not False:
            self.loss_conditional_metric.update_state(loss_conditional)
        self.loss_generator_metric.update_state(loss_generator)
        self.loss_distill_metric.update_state(loss_distill)
        results = {m.name: m.result() for m in self.train_metrics}
        return results

if __name__ == '__main__':
    from dataloader import dataloader
    from models.classifiers.LeNet_5 import LeNet_5_ReLU_MaxPool
    from models.classifiers.AlexNet import AlexNet
    from models.distillers.utils import add_fmap_output
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback

    def run_experiment_mnist_lenet5(pretrained_teacher:bool=False):
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
        BATCH_SIZE_TEACHER, BATCH_SIZE_DISTILL = 256, 500 # Change 512 to 500 for evenly distributed classes
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 10, 200
        OPTIMIZER_TEACHER = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
        # OPTIMIZER_GENERATOR = keras.optimizers.Adam(learning_rate=2e-2, epsilon=1e-8)
        # OPTIMIZER_STUDENT = keras.optimizers.Adam(learning_rate=2e-4, epsilon=1e-8)
        OPTIMIZER_GENERATOR = keras.optimizers.SGD(learning_rate=2e-2)
        OPTIMIZER_STUDENT = keras.optimizers.SGD(learning_rate=2e-4)
        COEFF_OH, COEFF_AC, COEFF_IE, COEFF_CN = 1, 0.1, 5, 1

        print(' Experiment 1: CDAFL on MNIST. Teacher: LeNet-5, student: LeNet-5-HALF '.center(80,'#'))

        ds, info = dataloader(
            dataset='mnist',
            resize=IMAGE_DIM[0:-1],
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
            batch_size_test=1024,
            with_info=True
        )
        class_names = info.features['label'].names

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

        generator = ConditionalDataFreeGenerator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            embed_dim=None,
            num_classes=NUM_CLASSES,
            onehot_input=True,
            dafl_batchnorm=True
        )
        generator.build()

        # Train one student with default data-free learning settings
        distiller = CDAFL(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=OPTIMIZER_STUDENT,
            optimizer_generator=OPTIMIZER_GENERATOR,
            onehot_loss_fn=False,
            activation_loss_fn=True,
            info_entropy_loss_fn=False,
            conditional_loss_fn=True,
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_oh=COEFF_OH,
            coeff_ac=COEFF_AC,
            coeff_ie=COEFF_IE,
            coeff_cn=COEFF_CN,
            confidence=None
        )

        csv_logger = keras.callbacks.CSVLogger(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.gif',
            postprocess_fn=lambda x:x*0.3081 + 0.1307,
            normalize=False,
            class_names=class_names,
            delete_png=False,
            save_freq=NUM_EPOCHS_DISTILL//50
        )
        distiller.fit(
            epochs=NUM_EPOCHS_DISTILL,
            callbacks=[csv_logger, gif_maker],
            validation_data=ds['test']
        )

    def run_experiment_mnist_alexnet(pretrained_teacher:bool=False):
        LATENT_DIM = 100
        IMAGE_DIM = [28, 28, 1]
        NUM_CLASSES = 10
        BATCH_SIZE_TEACHER, BATCH_SIZE_DISTILL = 256, 500 # Change 512 to 500 for evenly distributed classes
        NUM_EPOCHS_TEACHER, NUM_EPOCHS_DISTILL = 10, 200
        OPTIMIZER_TEACHER = keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
        # OPTIMIZER_GENERATOR = keras.optimizers.Adam(learning_rate=2e-2, epsilon=1e-8)
        # OPTIMIZER_STUDENT = keras.optimizers.Adam(learning_rate=2e-4, epsilon=1e-8)
        OPTIMIZER_GENERATOR = keras.optimizers.SGD(learning_rate=2e-2)
        OPTIMIZER_STUDENT = keras.optimizers.SGD(learning_rate=2e-4)
        COEFF_OH, COEFF_AC, COEFF_IE, COEFF_CN = 1, 0.1, 5, 1

        print(' Experiment 1: CDAFL on MNIST. Teacher: AlexNet, student: AlexNet-Half '.center(80,'#'))

        ds, info = dataloader(
            dataset='mnist',
            rescale='standardization',
            batch_size_train=BATCH_SIZE_TEACHER,
            batch_size_test=1024,
            with_info=True
        )
        class_names = info.features['label'].names

        # Teacher (AlexNet)
        teacher = AlexNet(input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        teacher.compile(
            metrics=['accuracy'], 
            optimizer=OPTIMIZER_TEACHER,
            loss=keras.losses.SparseCategoricalCrossentropy())
        teacher.build()

        if pretrained_teacher is True:
            teacher.load_weights('./pretrained/mnist/mean0_std1/AlexNet_9945.h5')
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
                validation_data=ds['test']
            )
            teacher.load_weights(filepath=f'./logs/{teacher.name}_best.h5')
        teacher.evaluate(ds['test'])
        teacher = add_fmap_output(model=teacher, fmap_layer='flatten')

        # Student (AlexNet-Half)
        student = AlexNet(half=True, input_dim=IMAGE_DIM, num_classes=NUM_CLASSES)
        student.build()
        student.compile(metrics='accuracy')
        student.evaluate(ds['test'])

        generator = ConditionalDataFreeGenerator(
            latent_dim=LATENT_DIM,
            image_dim=IMAGE_DIM,
            embed_dim=None,
            num_classes=NUM_CLASSES,
            onehot_input=True,
            dafl_batchnorm=True
        )
        generator.build()

        # Train one student with default data-free learning settings
        distiller = CDAFL(
            teacher=teacher, student=student, generator=generator)
        distiller.compile(
            optimizer_student=OPTIMIZER_STUDENT,
            optimizer_generator=OPTIMIZER_GENERATOR,
            onehot_loss_fn=False,
            activation_loss_fn=True,
            info_entropy_loss_fn=False,
            conditional_loss_fn=True,
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
            batch_size=BATCH_SIZE_DISTILL,
            num_batches=120,
            coeff_oh=COEFF_OH,
            coeff_ac=COEFF_AC,
            coeff_ie=COEFF_IE,
            coeff_cn=COEFF_CN,
            confidence=None
        )

        csv_logger = keras.callbacks.CSVLogger(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        gif_maker = MakeConditionalSyntheticGIFCallback(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.gif',
            postprocess_fn=lambda x:x*0.3081 + 0.1307,
            normalize=False,
            class_names=class_names,
            delete_png=False,
            save_freq=NUM_EPOCHS_DISTILL//50
        )
        distiller.fit(
            epochs=NUM_EPOCHS_DISTILL,
            callbacks=[csv_logger, gif_maker],
            validation_data=ds['test']
        )

    run_experiment_mnist_alexnet(pretrained_teacher=False)