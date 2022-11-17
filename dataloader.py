"""
Best practice summary (https://www.tensorflow.org/guide/data_performance#best_practice_summary)
Here is a summary of the best practices for designing performant TensorFlow input pipelines:
  - Use the prefetch transformation to overlap the work of a producer and consumer
  - Parallelize the data reading transformation using the interleave transformation
  - Parallelize the map transformation by setting the num_parallel_calls argument
  - Use the cache transformation to cache data in memory during the first epoch
  - Vectorize user-defined functions passed in to the map transformation
  - Reduce memory usage when applying the interleave, prefetch, and shuffle transformations

Augmentation snippets (put inside preprocess):
# First pad, then crop
x = tf.image.pad_to_bounding_box(x, offset_height=2, offset_width=2, target_height=36, target_width=36)
x = tf.image.random_crop(value=x, size=[tf.shape(x)[0], 32, 32, 3])
x = tf.image.random_flip_left_right(image=x)
"""

from typing import Literal, Tuple, Union, List, Callable, Any

import tensorflow as tf
import tensorflow_datasets as tfds

def compute_mean_std(dataset:str) -> Tuple[List[float], List[float]]:
    """Compute the channel-wise mean and standard deviation of a dataset, typically
    for rescaling.
    
    Args:
        `dataset`: Name of dataset.
    Returns:
        A tuple of `(mean, std)`.
    """    
    ds, info = tfds.load(dataset, as_supervised=True, with_info=True, data_dir='./datasets')
    num_channels = info.features['image'].shape[-1]
    num_examples = info.splits['train'].num_examples
    def preprocess(x, y):
        x = tf.cast(x, tf.float64)/255
        return x, y
    
    ds['train'] = ds['train'].map(preprocess).batch(num_examples).prefetch(1)
    x, _ = next(iter(ds['train']))

    mean = [None for i in range(num_channels)]
    std = [None for i in range(num_channels)]
    for channel in range(num_channels):
        mean[channel] = tf.math.reduce_mean(x[:, :, :, channel]).numpy()
        std[channel] = tf.math.reduce_std(x[:, :, :, channel]).numpy()

    return mean, std

def dataloader(dataset:str,
               augmentation_fn:Union[None, Callable[[Any], Any]]=None,
               resize:Union[None, Tuple[float, float]]=None,
               rescale:Union[Literal['standardization'], Tuple[float, float]]=(0, 1),
               batch_size_train:int=128,
               batch_size_test:int=1024,
               drop_remainder:bool=False,
               onehot_label:bool=False,
               with_info:bool=False):
    """Prepare data iterators from TensorFlow datasets.
    
    Args:
        `dataset`: Name of dataset.
        `resize`: Resizing dimension. Leave as `None` to skip resizing.
            Defaults to `None`.
        `rescale`: Rescaling method. Pass `'standardization'` to rescale to mean of
            0 and standard deviation of 1; or pass a tuple `(min, max)` to rescale
            to the range within [min, max].  Defaults to `(0, 1)`.
        `batch_size_train`: Batch size of training set. Defaults to `128`.
        `batch_size_test`: Batch size of test set. Defaults to `1024`.
        `drop_remainder`: Flag to drop the last batch. Defaults to `False`.
        `onehot_label`: Flag to produce one-hot label. Defaults to `False`.
        `with_info`: Flag to return dataset's info. Defaults to `False`.
    Returns:
        A dataset `ds`, or a dataset with its info `(ds, info)` when `with_info` is
            True.
    """                
    STANDARDIZATION_MEAN_STD = {
        'cifar10': (tf.constant([[[0.4914, 0.4822, 0.4465]]]), tf.constant([[[0.2470, 0.2435, 0.2616]]])),
        'cifar100': (tf.constant([[[0.5071, 0.4865, 0.4409]]]), tf.constant([[[0.2673, 0.2564, 0.2762]]])),
        'mnist': (tf.constant([[[0.1307]]]), tf.constant([[[0.3081]]])),
        'fashion_mnist': (tf.constant([[[0.2860]]]), tf.constant([[[0.3530]]])),
    }

    if rescale == 'standardization':
        mean, std = STANDARDIZATION_MEAN_STD[dataset]
    else:
        mean = -rescale[0]/(rescale[1] - rescale[0])
        std = 1/(rescale[1] - rescale[0])
    ds, info = tfds.load(dataset, as_supervised=True, with_info=True, data_dir='./datasets')
    num_classes = info.features['label'].num_classes

    if augmentation_fn is None:
        augmentation_fn = lambda x:x

    def preprocess(x, y):
        x = tf.cast(x, tf.float32)/255
        x = augmentation_fn(x)
        if resize is not None:
            x = tf.image.resize(images=x, size=resize)
        x = (x - mean)/std
        y = tf.cast(y, tf.int32)
        if onehot_label is True:
            y = tf.one_hot(indices=y, depth=num_classes)
        return x, y

    ds['train'] = (ds['train']
        # .interleave(dataset_generator_func, num_parallel_calls=tf.data.AUTOTUNE)  # Parallelize data reading
        .cache()                                                                    # Cache data
        .shuffle(50000)
        .batch(batch_size_train, drop_remainder=drop_remainder)                     # Vectorize your mapped function
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)                       # Reduce memory usage
        # .map(time_consuming_map, num_parallel_calls=tf.data.AUTOTUNE)             # Parallelize map transformation
        .prefetch(tf.data.AUTOTUNE))                                                # Overlap producer and consumer works
    ds['test'] = (ds['test']
        # .interleave(batch_size_test, num_parallel_calls=tf.data.AUTOTUNE)         # Parallelize data reading
        .cache()                                                                    # Cache data
        .batch(batch_size_test, drop_remainder=drop_remainder)                      # Vectorize your mapped function
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)                       # Reduce memory usage
        .prefetch(tf.data.AUTOTUNE))                                                # Overlap producer and consumer works
    
    if with_info is False:
        return ds
    elif with_info is True:
        return ds, info

if __name__ == '__main__':
    import time

    for dataset in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
        ds, info = dataloader(dataset, rescale=[-1, 1], batch_size_train=256, drop_remainder=False, with_info=True)
        start = time.time()
        for i in range(100):
            next(iter(ds['train']))
        end = time.time()
        print(dataset, end - start)