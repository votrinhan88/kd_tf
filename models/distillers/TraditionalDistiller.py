from typing import List
import tensorflow as tf
keras = tf.keras

class TraditionalDistiller(keras.Model):
    '''Traditional knowledge distillation scheme, training the student on both the
    transfer set and the soft targets produced by a pre-trained teacher.

    Args:
        `teacher`: To-be-trained student model. Should return logits.
        `student`: Pre-trained teacher model. Should return logits.

    Distilling the Knowledge in a Neural Network - Hinton et al. (2015)
    DOI: 10.48550/arXiv.1503.02531
    '''
    _name = 'TraditionalDistiller'
    def __init__(self,
                 teacher:keras.Model,
                 student:keras.Model,
                 **kwargs):
        """Initialize distiller.
        
        Args:
            `teacher`: To-be-trained student model. Should return logits.
            `student`: Pre-trained teacher model. Should return logits.
        """                 
        super().__init__(name=self._name, **kwargs)
        self.teacher = teacher
        self.student = student

    def compile(self,
                optimizer:keras.optimizers.Optimizer,
                metrics=None,
                distill_loss_fn:keras.losses.Loss=keras.losses.KLDivergence(),
                student_loss_fn:keras.losses.Loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                alpha:float=0.1,
                temperature:float=4.0,
                **kwargs):                
        """Compile distiller.
        
        Args:
            `optimizer`: Optimizer for student model.
            `metrics`: Additional metrics for model.
                Defaults to `None`.
                    Training phase: accuracy, student loss, distillation loss, total loss
                    Validation phase: accuracy, student loss
            `distill_loss_fn`: Loss function for distillation.
                Defaults to `keras.losses.KLDivergence()`.
            `student_loss_fn`: Loss function for student.
                Defaults to `keras.losses.SparseCategoricalCrossentropy(from_logits=True)`.
            `alpha`: weight assigned to student loss. Correspondingly, weight assigned
            to distillation loss is `1 - alpha`.
                Defaults to `0.1`.
            `temperature`: Temperature for softening probability distributions. Larger
            temperature gives softer distributions.
                Defaults to `4.0`.
        """
        super().compile(metrics=metrics, **kwargs)
        self.optimizer = optimizer
        self.student_loss_fn = student_loss_fn
        self.distill_loss_fn = distill_loss_fn
        self.alpha = alpha
        self.temperature = temperature

        # Metrics
        self.loss_student_metric = keras.metrics.Mean(name='loss_student')
        self.loss_distill_metric = keras.metrics.Mean(name='loss_distill')
        self.loss_total_metric = keras.metrics.Mean(name='loss_total')
        self.accuracy_metric = keras.metrics.Accuracy(name='accuracy')
    
    @property
    def train_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring training step.
        
        Returns:
            List of training metrics.
        """
        return [self.loss_student_metric,
                self.loss_distill_metric,
                self.loss_total_metric,
                self.accuracy_metric]
    
    @property
    def val_metrics(self) -> List[keras.metrics.Metric]:
        """Metrics monitoring validation step.
        
        Returns:
            List of validation metrics.
        """
        return [self.loss_student_metric,
                self.accuracy_metric]

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

        results = {m.name: m.result() for m in self.metrics}
        results.update({m.name: m.result() for m in self.train_metrics})
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

        results = {m.name: m.result() for m in self.metrics}
        results.update({m.name: m.result() for m in self.train_metrics})
        return results

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.classifiers.LeNet_5 import LeNet_5
    from dataloaders import get_MNIST

    # Configs
    CONFIG_PRETRAINED_BASELINE = True
    # Hyperparameters
    NUM_EPOCHS = 10
    ALPHA = 0.5
    TEMPERATURE = 10
    # Get dataset CIFAR10
    ds = get_MNIST()
    # Optimizer & scheduler
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Pre-trained teacher
    teacher = LeNet_5()
    teacher.compile(metrics=['accuracy'])
    teacher.load_weights('./pretrained/mnist/LeNet-5_tanh_AvgPool_9906.h5')
    teacher.evaluate(ds['test'])

    # TODO: Finish `__main__()`
    # Baseline (same size as student)
    baseline = LeNet_5(half=True)
    baseline.compile(optimizer, loss_fn, metrics=['accuracy'])
    if CONFIG_PRETRAINED_BASELINE == True:
        baseline.load_weights('./pretrained/mnist/LeNet-5-HALF_tanh_AvgPool_9867.h5')
    elif CONFIG_PRETRAINED_BASELINE == False:
        best_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f'./checkpoints/{baseline.name}_{int(ALPHA*10):02}_{TEMPERATURE:02}_baseline.ckpt',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=True)
        csv_logger = tf.keras.callbacks.CSVLogger(
            f'./logs/{baseline.name}_baseline.csv',
            append=True)
        baseline.fit(
            ds['train'],
            batch_size=BATCH_SIZE,
            steps_per_epoch = 50000//128, 
            epochs=200,
            validation_data=ds['test'],
            verbose=1,
            callbacks=[scheduler_callback, csv_logger, best_callback])
        baseline.load_weights(f'./checkpoints/{baseline.name}_baseline.ckpt')
    baseline.evaluate(ds['test'])
    
    # Knowledge distillation for students
    student = resnet_v1(20)
    student.compile(optimizer, loss_fn, metrics=['accuracy'])
    best_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'./checkpoints/{student.name}_{int(ALPHA*10):02}_{TEMPERATURE:02}_student_best.ckpt',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'./logs/{student.name}_{int(ALPHA*10):02}_{TEMPERATURE:02}_student.csv',
        append=True)

    distiller = TraditionalDistiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optimizer,
        metrics=['accuracy'],
        student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        alpha = ALPHA,
        temperature = TEMPERATURE)
    distiller.fit(
        ds['train'],
        batch_size=BATCH_SIZE,
        steps_per_epoch = 50000//128, 
        epochs=200,
        validation_data=ds['test'],
        verbose=1,
        callbacks=[scheduler_callback, csv_logger, best_callback])
    
    student.load_weights(
        f'./checkpoints/{student.name}_{int(ALPHA*10):02}_{TEMPERATURE:02}_student_best.ckpt')
    student.evaluate(ds['test'])