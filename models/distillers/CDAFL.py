import warnings
from typing import List, Union, Tuple
import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.distillers.DataFreeDistiller import DataFreeDistiller
else:
    from .DataFreeDistiller import DataFreeDistiller, DataFreeGenerator

class ConditionalDataFreeGenerator(DataFreeGenerator):
    """Review GAN to CGAN to modify.
    """
    pass

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
    pass