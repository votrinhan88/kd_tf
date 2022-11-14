"""
latent_dim = 2 
num_epochs = 600

Generate 500 images
CIFAR-10: each class has 50 generated images
CIFAR-100: each class has 5 generated images
save the generated images and their labels to numpy file
"""

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

import numpy as np
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds

from models.GANs.conditional_gan import (
    ConditionalDiscriminatorStack,
    ConditionalGeneratorStack,
    ConditionalGenerativeAdversarialNetwork
)

from models.GANs.utils import (
    MakeConditionalSyntheticGIFCallback,
    MakeInterpolateSyntheticGIFCallback
)

# Load Fewshot CIFAR-10
with open("./datasets/fewshot_cgan/mnist_x.npz", "rb") as f:
    X_train = np.load(f)
    X_train = tf.expand_dims(input=X_train, axis=-1)
with open("./datasets/fewshot_cgan/mnist_y.npz", "rb") as f:
    y_train = np.load(f)
    y_train = np.squeeze(y_train)

ds = {'train': None, 'test': None}
ds['train'] = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds['test']:tf.data.Dataset = tfds.load('mnist', as_supervised=True)['test']
def preprocess(x, y):
    x = tf.cast(x, tf.float32)/255.
    x = (x - 0.5)/0.5
    y = tf.cast(y, tf.int32)
    y = tf.one_hot(indices=y, depth=10)
    return x, y
ds['train'] = ds['train'].map(preprocess).shuffle(X_train.shape[0]).batch(X_train.shape[0], drop_remainder=True).prefetch(1)
ds['test'] = ds['test'].map(preprocess).batch(500, drop_remainder=True).prefetch(1)

cgen = ConditionalGeneratorStack(
    latent_dim=2,
    image_dim=[28, 28, 1],
    base_dim=[7, 7, None],
    num_classes=10
)
cgen.build()

cdisc = ConditionalDiscriminatorStack(
    image_dim=[28, 28, 1],
    num_classes=10
)
cdisc.build()

cgan = ConditionalGenerativeAdversarialNetwork(
    generator=cgen, discriminator=cdisc, embed_dim=-1
)
cgan.build()
cgan.summary(line_length=120, expand_nested=True)

cgan.compile(
    optimizer_gen=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5),
    optimizer_disc=keras.optimizers.Adam(learning_rate=0.0003, beta_1=0.5)
)
csv_logger = keras.callbacks.CSVLogger(
    filename=f'./logs/fewshot_cgan_mnist.csv',
    append=True)

gif_maker = MakeConditionalSyntheticGIFCallback(
    filename=f'./logs/fewshot_cgan_mnist.gif',
    postprocess_fn=lambda x:(x+1)/2,
    save_freq=10
)

cgan.fit(
    x=ds['train'],
    epochs=600,
    callbacks=[csv_logger, gif_maker],
    validation_data=ds['test'],
)

def make_cgan_inputs(latent_dim:int=2, num_classes:int=10, num_examples_per_class:int=50, seed:int=17, onehot_input:bool=True):
    latents = tf.random.normal(
        shape=(num_classes*num_examples_per_class, latent_dim),
        seed=seed
    )
    labels = tf.range(start=0, limit=num_classes, delta=1)
    labels = tf.repeat(labels, repeats=num_examples_per_class, axis=0)
    if onehot_input is True:
        labels = tf.one_hot(indices=labels, depth=num_classes)

    return latents, labels

latents, labels = make_cgan_inputs(
    latent_dim=2,
    num_classes=10,
    num_examples_per_class=50,
    onehot_input=True)

x_synth = cgan.generator([latents, labels])
y_synth = tf.math.argmax(labels, axis=1)

with open("./logs/mnist_x_synth.npz", "wb") as f:
    np.save(f, x_synth)
with open("./logs/mnist_y_synth.npz", "wb") as f:
    np.save(f, y_synth)