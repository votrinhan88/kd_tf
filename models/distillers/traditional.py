import tensorflow as tf
keras = tf.keras

class DistillerTraditional(keras.Model):
    '''
    Distilling the Knowledge in a Neural Network
    https://doi.org/10.48550/arXiv.1503.02531
    '''
    def __init__(self, student:keras.Model, teacher:keras.Model):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer:keras.optimizers.Optimizer,
        metrics,
        student_loss_fn:keras.losses.Loss,
        distillation_loss_fn:keras.losses.Loss,
        alpha:float=0.1,
        temperature:float=4,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

if __name__ == '__main__':
    # Change path
    import os, sys
    sys.path.append(os.path.realpath(os.path.dirname(os.path.dirname(os.path.dirname(sys.argv[0])))))

    import tensorflow_datasets as tfds
    from models.resnet import resnet_v1
    from dataloaders import get_CIFAR10

    # Configs
    CONFIG_PRETRAINED_BASELINE = True
    # Hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    ALPHA = 0.5
    TEMPERATURE = 10
    # Get dataset CIFAR10
    ds = get_CIFAR10()
    # Optimizer & scheduler
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    def scheduler(epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        return lr
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler(0))
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Pre-trained teacher
    teacher = resnet_v1(56)
    teacher.compile(optimizer, loss_fn, metrics=['accuracy'])
    teacher.load_weights('./pretrained/CIFAR10/ResNet56_v1.ckpt')
    teacher.evaluate(ds['test'])

    # Baseline (same size as student)
    baseline = resnet_v1(20)
    baseline.compile(optimizer, loss_fn, metrics=['accuracy'])
    if CONFIG_PRETRAINED_BASELINE == True:
        baseline.load_weights('./pretrained/CIFAR10/ResNet20_v1.ckpt')
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
        save_weights_only = True)
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'./logs/{student.name}_{int(ALPHA*10):02}_{TEMPERATURE:02}_student.csv',
        append=True)

    distiller = DistillerTraditional(student=student, teacher=teacher)
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
    
    student.load_weights(f'./checkpoints/{student.name}_{int(ALPHA*10):02}_{TEMPERATURE:02}_student_best.ckpt')
    student.evaluate(ds['test'])