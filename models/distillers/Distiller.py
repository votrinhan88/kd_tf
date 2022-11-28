from typing import List, Union, Tuple, Callable
import tensorflow as tf
keras = tf.keras
from keras.utils import io_utils


class Distiller(keras.Model):
    '''Traditional knowledge distillation scheme, training the student on both the
    transfer set and the soft targets produced by a pre-trained teacher.

    Args:
        `teacher`: To-be-trained student model. Should return logits.
        `student`: Pre-trained teacher model. Should return logits.
        `input_dim`: Dimension of input images, leave as `None` to be parsed from
            student. Defaults to `None`.

    Kwargs:
        Additional keyword arguments passed to `keras.Model.__init__`.

    Distilling the Knowledge in a Neural Network - Hinton et al. (2015)
    DOI: 10.48550/arXiv.1503.02531
    '''
    _name = 'Distiller'
    def __init__(self,
                 teacher:keras.Model,
                 student:keras.Model,
                 image_dim:Union[None, List[int]]=None,
                 **kwargs):
        """Initialize distiller.
        
        Args:
            `teacher`: To-be-trained student model. Should return logits.
            `student`: Pre-trained teacher model. Should return logits.
            `image_dim`: Dimension of input images, leave as `None` to be parsed from
                student. Defaults to `None`.

        Kwargs:
            Additional keyword arguments passed to `keras.Model.__init__`.
        """        
        super().__init__(name=self._name, **kwargs)
        self.teacher = teacher
        self.student = student
        self.image_dim = image_dim
        
        if image_dim is None:
            self.image_dim:int = self.student.input_dim
        elif image_dim is not None:
            self.image_dim = image_dim

    def compile(self,
                optimizer:keras.optimizers.Optimizer,
                distill_loss_fn:keras.losses.Loss=keras.losses.KLDivergence(),
                student_loss_fn:keras.losses.Loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                alpha:float=0.1,
                temperature:float=4.0,
                **kwargs):
        """Compile distiller.
        
        Args:
            `optimizer`: Optimizer for student model.
            `distill_loss_fn`: Loss function for distillation from teacher.
                Defaults to `keras.losses.KLDivergence()`.
            `student_loss_fn`: Loss function for learning from training data.
                Defaults to `keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
            `alpha`: weight assigned to student loss. Correspondingly, weight assigned
                to distillation loss is `1 - alpha`. Defaults to `0.1`.
            `temperature`: Temperature for softening probability distributions. Larger
                temperature gives softer distributions. Defaults to `4.0`.
        
        Kwargs:
            Additional keyword arguments passed to `keras.Model.compile`.
        """
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temperature = temperature

        # Metrics
        self.loss_student_metric = keras.metrics.Mean(name='loss_st')
        self.loss_distill_metric = keras.metrics.Mean(name='loss_dt')
        self.loss_total_metric = keras.metrics.Mean(name='loss')
        self.accuracy_metric = keras.metrics.Accuracy(name='accuracy')
    
    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring training step.
        
        Returns:
            List of training metrics.
        """
        train_metrics = self.metrics
        train_metrics.extend([
            self.loss_student_metric,
            self.loss_distill_metric,
            self.loss_total_metric,
            self.accuracy_metric
        ])
        return train_metrics
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring validation step.
        
        Returns:
            List of validation metrics.
        """
        val_metrics = self.metrics
        val_metrics.extend([
            self.loss_student_metric,
            self.accuracy_metric
        ])
        return val_metrics

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_logits = self.teacher(x, training=False)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.student.trainable_variables)

            # Forward pass of student
            student_logits = self.student(x, training=True)
            student_prob = tf.nn.softmax(student_logits, axis=1)

            # Compute losses
            loss_student = self.student_loss_fn(y, student_prob)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            loss_distill = self.temperature**2 * self.distill_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1),
            )

            loss_total = self.alpha*loss_student + (1 - self.alpha)*loss_distill

        # Back-propagation
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss_total, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics, configured in `compile()`.
        self.loss_student_metric.update_state(loss_student)
        self.loss_distill_metric.update_state(loss_distill)
        self.loss_total_metric.update_state(loss_total)
        self.accuracy_metric.update_state(y_true=y, y_pred=student_prob)

        results = {m.name: m.result() for m in self.train_metrics}
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

       # Forward pass of student
        student_logits = self.student(x, training=False)
        student_prob = tf.nn.softmax(student_logits, axis=1)

        # Compute losses
        loss_student = self.student_loss_fn(y, student_prob)

        # Update the metrics, configured in `compile()`.
        self.loss_student_metric.update_state(loss_student)
        self.accuracy_metric.update_state(y_true=y, y_pred=student_prob)

        results = {m.name: m.result() for m in self.train_metrics}
        return results

    def build(self):
        super().build(input_shape=[None, *self.image_dim])

    def summary(self, as_functional:bool=False, **kwargs):
        """Prints a string summary of the network.

        Args:
            `as_functional`: Flag to print from a dummy functional model.
                Defaults to `False`.

        Kwargs:
            Additional keyword arguments passed to `keras.Model.summary`.
        """
        inputs = keras.layers.Input(shape=self.image_dim)
        outputs = self.call(inputs)

        if as_functional is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'teacher_class':self.teacher.__class__,
            'teacher':self.teacher.get_config(),
            'student_class':self.student.__class__,
            'student':self.student.get_config(),
            'image_dim':self.image_dim,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        teacher = config['teacher_class'].from_config(config['teacher'])
        teacher.build()
        student = config['student_class'].from_config(config['student'])
        student.build()

        config.update({
            'teacher':teacher,
            'student':student
        })
        for key in ['teacher_class', 'student_class']:
            config.pop(key, None)
        return super().from_config(config, custom_objects)

if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    import tensorflow_datasets as tfds

    from models.classifiers.LeNet_5 import LeNet_5
    from models.classifiers.HintonNet import HintonNet
    from models.distillers.utils import CSVLogger_custom, ValidationFreqPrinter

    def run_experiment_mnist_hintonnet():
        IMAGE_DIM = [28, 28, 1]
        NUM_CLASSES = 10
        BATCH_SIZE = 100
        HIDDEN_LAYERS_TEACHER, HIDDEN_LAYERS_STUDENT = [1200, 1200], [800, 800]
        NUM_STEPS = 3000
        TEMPERATURE = 8
        ALPHA = 0.5

        class MomentumLearningRateSchedule():
            def __init__(self,
                         init_momentum:float=0.5,
                         stop_momentum:float=0.99,
                         stop_step:int=500,
                         init_learning_rate:float=10.0,
                         exponential_base:float=0.998):
                self.init_momentum = init_momentum
                self.stop_momentum = stop_momentum
                self.stop_step = stop_step
                self.init_learning_rate = init_learning_rate
                self.exponential_base = exponential_base
            
            def __call__(self, step:int):
                # Compute momentum
                if step < self.stop_step:
                    momentum = (1 - step/self.stop_step)*self.init_momentum + \
                                   (step/self.stop_step)*self.stop_momentum
                elif step >= self.stop_step:
                    momentum = self.stop_momentum
                # Compute learning rate
                learning_rate = (1 - momentum)*(self.exponential_base**step)
                return momentum, learning_rate

        class MomentumLearningRateSchedulerCustom(keras.callbacks.Callback):
            def __init__(self,
                         schedule:Callable[[int], Tuple[float, float]],
                         optimizer_name:str='optimizer',
                         verbose:int=0,
                         **kwargs):
                super().__init__(**kwargs)
                self.optimizer_name = optimizer_name
                self.schedule = schedule
                self.verbose = verbose

            def on_train_begin(self, logs=None):
                self.optimizer:keras.optimizers.Optimizer = self.model.__getattribute__(self.optimizer_name)
                return super().on_train_begin(logs)

            def on_epoch_begin(self, epoch:int, logs=None):
                old_mmt = self.optimizer.momentum.read_value()
                old_lr = self.optimizer.lr.read_value()
                new_mmt, new_lr = self.schedule(epoch)
                self.optimizer.momentum.assign(new_mmt)
                self.optimizer.lr.assign(new_lr)
                if (self.verbose > 0):
                    if (new_lr != old_lr):
                        io_utils.print_msg(
                            f'Learning rate of `{self.optimizer_name}` has been changed to '
                            f'{new_lr}.'
                        )
                    if (new_mmt != old_mmt):
                        io_utils.print_msg(
                            f'Momentum of `{self.optimizer_name}` has been changed to '
                            f'{new_mmt}.'
                        )
                return super().on_epoch_begin(epoch, logs)

        OPTIMIZER = keras.optimizers.SGD()
        
        print(' Standard knowledge distillation on MNIST '.center(80,'#'))
        def augmentation_fn(x):
            x = tf.pad(tensor=x, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode='SYMMETRIC')
            x = tf.image.random_crop(value=x, size=[tf.shape(x)[0], *IMAGE_DIM])
            return x
        
        ds, info = tfds.load('mnist', as_supervised=True, with_info=True, data_dir='./datasets')
        num_examples = info.splits['train'].num_examples
        def train_preprocess(x, y):
            x = tf.cast(x, tf.float32)/255
            x = augmentation_fn(x)
            y = tf.cast(y, tf.int32)
            return x, y
        def test_preprocess(x, y):
            x = tf.cast(x, tf.float32)/255
            y = tf.cast(y, tf.int32)
            return x, y
        ds['train'] = (ds['train']
            .cache()
            .repeat()
            .shuffle(num_examples)
            .batch(BATCH_SIZE)                     
            .map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)                 
            .prefetch(tf.data.AUTOTUNE))                                                
        ds['test'] = (ds['test']
            .cache()
            .batch(1000)
            .map(test_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))   

        # Teacher (LeNet-5)
        teacher = HintonNet(
            input_dim=IMAGE_DIM,
            hidden_layers=HIDDEN_LAYERS_TEACHER,
            num_classes=NUM_CLASSES,
            return_logits=True,
        )
        teacher.compile(
            metrics=['accuracy'], 
            optimizer=OPTIMIZER,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        teacher.build()

        csv_logger = CSVLogger_custom(
            filename=f'./logs/{teacher.name}_teacher.csv',
            append=True
        )
        best_callback = keras.callbacks.ModelCheckpoint(
            filepath=f'./logs/{teacher.name}_teacher_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
        scheduler = MomentumLearningRateSchedulerCustom(
            schedule=MomentumLearningRateSchedule(
                init_momentum=0.5,
                stop_momentum=0.99,
                stop_step=500,
                init_learning_rate=10,
                exponential_base=0.998),
            optimizer_name='optimizer')
        printer = ValidationFreqPrinter(validation_freq=100)
        teacher.fit(
            x=ds['train'],
            steps_per_epoch=1,
            epochs=1,
            callbacks=[csv_logger, scheduler, printer],
            validation_data=ds['test'],
            verbose=0,
        )
        teacher.fit(
            x=ds['train'],
            steps_per_epoch=1,
            epochs=NUM_STEPS,
            initial_epoch=1,
            callbacks=[best_callback, csv_logger, printer, scheduler],
            validation_data=ds['test'],
            validation_freq=100,
            verbose=0,
        )
        teacher.load_weights(filepath=f'./logs/{teacher.name}_teacher_best.h5')
        teacher.evaluate(ds['test'])

        # Student (LeNet-5-HALF)
        student = HintonNet(
            input_dim=IMAGE_DIM,
            hidden_layers=HIDDEN_LAYERS_STUDENT,
            num_classes=NUM_CLASSES,
            return_logits=True,
        )
        student.compile(
            metrics=['accuracy'], 
            optimizer=OPTIMIZER.from_config(OPTIMIZER.get_config()),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        student.build()
        
        csv_logger = CSVLogger_custom(
            filename=f'./logs/{student.name}_student.csv',
            append=True
        )
        best_callback = keras.callbacks.ModelCheckpoint(
            filepath=f'./logs/{student.name}_student_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
        scheduler = MomentumLearningRateSchedulerCustom(
            schedule=MomentumLearningRateSchedule(
                init_momentum=0.5,
                stop_momentum=0.99,
                stop_step=500,
                init_learning_rate=10,
                exponential_base=0.998),
            optimizer_name='optimizer')
        printer = ValidationFreqPrinter(validation_freq=100)
        student.fit(
            x=ds['train'],
            steps_per_epoch=1,
            epochs=1,
            callbacks=[csv_logger, scheduler, printer],
            validation_data=ds['test'],
            verbose=0,
        )
        student.fit(
            x=ds['train'],
            steps_per_epoch=1,
            epochs=NUM_STEPS,
            initial_epoch=1,
            callbacks=[best_callback, csv_logger, printer, scheduler],
            validation_data=ds['test'],
            validation_freq=100,
            verbose=0,
        )
        student.load_weights(filepath=f'./logs/{student.name}_student_best.h5')
        student.evaluate(ds['test'])

        # Standard knowledge distillation
        distiller = Distiller(teacher=teacher, student=student)
        distiller.build()
        distiller.summary(as_functional=True, expand_nested=True, line_length=120)
        distiller.compile(
            optimizer=OPTIMIZER.from_config(OPTIMIZER.get_config()),
            distill_loss_fn=keras.losses.KLDivergence(),
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            alpha=ALPHA,
            temperature=TEMPERATURE
        )

        csv_logger = CSVLogger_custom(
            filename=f'./logs/{distiller.name}_{student.name}_mnist.csv',
            append=True
        )
        scheduler = MomentumLearningRateSchedulerCustom(
            schedule=MomentumLearningRateSchedule(
                init_momentum=0.5,
                stop_momentum=0.99,
                stop_step=500,
                init_learning_rate=10,
                exponential_base=0.998),
            optimizer_name='optimizer')
        printer = ValidationFreqPrinter(validation_freq=100)
        distiller.fit(
            x=ds['train'],
            steps_per_epoch=1,
            epochs=1,
            callbacks=[csv_logger, scheduler, printer],
            validation_data=ds['test'],
            verbose=0,
        )
        distiller.fit(
            x=ds['train'],
            steps_per_epoch=1,
            epochs=NUM_STEPS,
            initial_epoch=1,
            callbacks=[csv_logger, scheduler, printer],
            validation_data=ds['test'],
            validation_freq=100,
            verbose=0,
        )
    run_experiment_mnist_hintonnet()