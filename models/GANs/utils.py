import warnings
from typing import Callable, Union, List, Any
import os
import glob

import tensorflow as tf
keras = tf.keras
from matplotlib.pyplot import subplots, close
import PIL

class MakeSyntheticGIFCallback(keras.callbacks.Callback):
    """Callback to generate synthetic images, typically used with a Generative
    Adversarial Network.
    
    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/GAN.gif'`.
        `nrows`: Number of rows in subplot figure. Defaults to `5`.
        `ncols`: Number of columns in subplot figure. Defaults to `5`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training. Defaults to `True`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png` after
            training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds.
            Defaults to `5000`.
    """
    def __init__(self,
                 filename:str='./logs/GAN.gif',
                 nrows:int=5,
                 ncols:int=5,
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 keep_noise:bool=True,
                 delete_png:bool=True,
                 duration:float=5000,
                 **kwargs):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/GAN.gif'`.
            `nrows`: Number of rows in subplot figure. Defaults to `5`.
            `ncols`: Number of columns in subplot figure. Defaults to `5`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """
        super().__init__(**kwargs)
        self.filename = filename
        self.nrows = nrows
        self.ncols = ncols
        self.postprocess_fn = postprocess_fn
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.keep_noise = keep_noise
        self.delete_png = delete_png
        self.duration = duration

        self.path_png_folder = self.filename[0:-4] + '_png'

    def on_train_begin(self, logs=None):
        self.handle_args()
        # Renew/create folder containing PNG files
        if os.path.isdir(self.path_png_folder):
            for png in glob.glob(f'{self.path_png_folder}/*.png'):
                os.remove(png)
        else:
            os.mkdir(self.path_png_folder)
        self.precompute_inputs()
        return super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        x_synth = self.synthesize_images()
        self.make_figure(x_synth, epoch)
        return super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        # Make GIF
        path_png = f'{self.path_png_folder}/*.png'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(path_png))]
        img.save(
            fp=self.filename,
            format='GIF',
            append_images=imgs,
            save_all=True,
            duration=self.duration/(len(imgs) + 1),
            loop=0)

        if self.delete_png is True:
            for png in glob.glob(path_png):
                os.remove(png)
            os.rmdir(self.path_png_folder)

        return super().on_train_end(logs)

    def handle_args(self):
        """Handle input arguments to callback, as some are not accessible in __init__().
        """
        if self.postprocess_fn is None:
            self.postprocess_fn = lambda x:x

        if self.latent_dim is None:
            self.latent_dim:int = self.model.latent_dim

        if self.image_dim is None:
            self.image_dim:int = self.model.image_dim

    def precompute_inputs(self):
        """Pre-compute inputs to feed to the generator. Eg: latent noise.
        """
        batch_size = self.nrows*self.ncols
        self.latent_noise = tf.random.normal(shape=(batch_size, self.latent_dim))

    def synthesize_images(self) -> tf.Tensor:
        """Produce synthetic images with the generator.
        
        Returns:
            A batch of synthetic images.
        """
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            self.latent_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        x_synth = self.model.generator(self.latent_noise)
        x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def make_figure(self, x_synth:tf.Tensor, epoch:int):
        """Tile the synthetic images into a nice grid, then make and save a figure at
        the given epoch.
        
        Args:
            `x_synth`: A batch of synthetic images.
            `epoch`: Current epoch.
        """
        # TODO:
        #   - Zero-padding currently only works for already normalized range.
        #   - Label for each category (it is obvious with MNIST, but might not
        #       for other datasets)
        #   - Update `cmap` for multi-channel pictures (eg CIFAR)
        fig, ax = subplots(constrained_layout=True, figsize=(self.ncols, 0.5 + self.nrows))
        fig.suptitle(f'{self.model.name} - Epoch {epoch}')

        # Tile images into a grid
        x = x_synth
        x = tf.pad(x, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT', constant_values=1)
        x = tf.reshape(x, shape=[self.nrows, self.ncols, *x.shape[1:]])
        x = tf.concat(tf.unstack(x, axis=0), axis=1)
        x = tf.concat(tf.unstack(x, axis=0), axis=1)
        x = tf.image.crop_to_bounding_box(
            image=x,
            offset_height=1, offset_width=1,
            target_height=x.shape[0]-1, target_width=x.shape[1]-1
        )
        ax.axis('off')
        ax.imshow(x.numpy().squeeze(axis=-1), cmap='gray')
        fig.savefig(f"{self.path_png_folder}/{self.model.name}_epoch_{epoch:04d}.png")
        close(fig)

class MakeConditionalSyntheticGIFCallback(MakeSyntheticGIFCallback):
    """Callback to generate synthetic images, typically used with a Conditional
    Generative Adversarial Network.

    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/GAN.gif'`.
        `target_classes`: The conditional target classes to make synthetic images,
            also is the columns in the figure. Leave as `None` to include all
            classes. Defaults to `None`.
        `num_samples_per_class`: Number of sample per class, also is the number of
            rows in the figure. Defaults to `5`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `num_classes`: Number of classes, leave as `None` to be parsed from model.
            Defaults to `None`.
        `onehot_input`: Flag to indicate whether the GAN model/generator receives
            one-hot or label encoded target classes, leave as `None` to be parsed
            from model. Defaults to `None`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training. Defaults to `True`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png` after
            training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds.
            Defaults to `5000`.
    """
    def __init__(self,
                 filename:str='./logs/GAN.gif',
                 target_classes:Union[None, List[int]]=None,
                 num_samples_per_class:int=5,
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 num_classes:Union[None, int]=None,
                 onehot_input:Union[None, bool]=None,
                 keep_noise:bool=True,
                 delete_png:bool=True,
                 duration:float=5000,
                 **kwargs):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/GAN.gif'`.
            `target_classes`: The conditional target classes to make synthetic images,
                also is the columns in the figure. Leave as `None` to include all
                classes. Defaults to `None`.
            `num_samples_per_class`: Number of sample per class, also is the number of
                rows in the figure. Defaults to `5`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from model.
                Defaults to `None`.
            `onehot_input`: Flag to indicate whether the GAN model/generator receives
                one-hot or label encoded target classes, leave as `None` to be parsed
                from model. Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """                 
        super(MakeConditionalSyntheticGIFCallback, self).__init__(
            filename      =filename,
            nrows         =None,
            ncols         =None,
            postprocess_fn=postprocess_fn,
            latent_dim    =latent_dim,
            image_dim     =image_dim,
            keep_noise    =keep_noise,
            delete_png    =delete_png,
            duration      =duration,
            **kwargs
        )
        self.target_classes = target_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.onehot_input = onehot_input

    def handle_args(self):
        super(MakeConditionalSyntheticGIFCallback, self).handle_args()
        if self.num_classes is None:
            self.num_classes:int = self.model.num_classes

        if self.onehot_input is None:
            self.onehot_input:bool = self.model.onehot_input

        if self.target_classes is None:
            self.target_classes = [label for label in range(self.num_classes)]
        
        self.nrows = self.num_samples_per_class
        self.ncols = len(self.target_classes)

    def precompute_inputs(self):
        super(MakeConditionalSyntheticGIFCallback, self).precompute_inputs()

        self.label = tf.tile(
            input=tf.constant(self.target_classes),
            multiples=[self.nrows]
        )
        if self.onehot_input is True:
            self.label = tf.one_hot(indices=self.label, depth=self.num_classes, axis=1)

    def synthesize_images(self):
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            self.latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
                    
        x_synth = self.model.generator([self.latent_noise, self.label])
        x_synth = self.postprocess_fn(x_synth)
        return x_synth

class MakeInterpolateSyntheticGIFCallback(MakeSyntheticGIFCallback):
    """Callback to generate synthetic images, interpolated between the classes of a
    Conditional Generative Adversarial Network.
    
    The callback can only work with models receiving one-hot encoded inputs. It
    will make figures at the end of the last epoch.
    
    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/GAN_itpl.gif'`.
        `start_classes`: Classes at the start of interpolation along the rows, leave
            as `None` to include all classes. Defaults to `None`.
        `end_classes`: Classes at the end of interpolation along the columns, leave
            as `None` to include all classes. Defaults to `None`.
        `num_interpolate`: Number of interpolation. Defaults to `21`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `num_classes`: Number of classes, leave as `None` to be parsed from model.
            Defaults to `None`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training. Defaults to `True`.
        `delete_png`: Flag to delete PNG files and folder at `filename/png` after
            training. Defaults to `True`.
        `duration`: Duration of the generated GIF in milliseconds.
            Defaults to `5000`.
    """
    def __init__(self,
                 filename:str='./logs/GAN_itpl.gif',
                 start_classes:List[int]=None,
                 end_classes:List[int]=None,
                 num_interpolate:int=21,
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 num_classes:Union[None, int]=None,
                 keep_noise:bool=True,
                 delete_png:bool=True,
                 duration:float=5000,
                 **kwargs):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/GAN_itpl.gif'`.
            `start_classes`: Classes at the start of interpolation along the rows, leave
                as `None` to include all classes. Defaults to `None`.
            `end_classes`: Classes at the end of interpolation along the columns, leave
                as `None` to include all classes. Defaults to `None`.
            `num_interpolate`: Number of interpolation. Defaults to `21`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `num_classes`: Number of classes, leave as `None` to be parsed from model.
                Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """
        assert num_interpolate > 2, (
            '`num_interpolate` (including the left and right classes) must be' +
            ' larger than 2.'
        )
        super().__init__(
            filename      =filename,
            nrows         =None,
            ncols         =None,
            postprocess_fn=postprocess_fn,
            latent_dim    =latent_dim,
            image_dim     =image_dim,
            keep_noise    =keep_noise,
            delete_png    =delete_png,
            duration      =duration,
            **kwargs
        )
        self.start_classes = start_classes
        self.end_classes = end_classes
        self.num_interpolate = num_interpolate
        self.num_classes = num_classes
    
    def on_epoch_end(self, epoch, logs=None):
        # Deactivate on_epoch_end()
        return keras.callbacks.Callback.on_epoch_end(self, epoch, logs)
    
    def on_train_end(self, logs=None):
        # One-hot encode start- and end-classes
        start_oh = tf.one_hot(indices=self.start_classes, depth=self.num_classes)
        start_oh = tf.expand_dims(input=start_oh, axis=1)
        start_oh = tf.broadcast_to(input=start_oh, shape=[self.nrows, self.ncols, self.num_classes])

        end_oh = tf.one_hot(indices=self.end_classes, depth=self.num_classes)
        end_oh = tf.expand_dims(input=end_oh, axis=0)
        end_oh = tf.broadcast_to(input=end_oh, shape=[self.nrows, self.ncols, self.num_classes])

        # Interpolate from start- to end-classes
        interpolate_ratios = tf.cast(tf.linspace(start=0, stop=1, num=self.num_interpolate), dtype=tf.float32).numpy().tolist()
        for ratio in interpolate_ratios:
            label = ((1-ratio)*start_oh + ratio*end_oh)
            self.label = tf.concat(tf.unstack(label), axis=0)
            x_synth = self.synthesize_images()
            self.make_figure(x_synth, ratio)

        # Make GIF
        path_png = f'{self.path_png_folder}/*.png'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(path_png))]
        img.save(
            fp=self.filename,
            format='GIF',
            append_images=imgs,
            save_all=True,
            duration=self.duration/(len(imgs) + 1),
            loop=0)

        if self.delete_png is True:
            for png in glob.glob(path_png):
                os.remove(png)
            os.rmdir(self.path_png_folder)

        return keras.callbacks.Callback.on_train_end(self, logs)

    def handle_args(self):
        super().handle_args()
        if self.model.onehot_input is None:
            warnings.warn(
                f'Model {self.model.name} does not have attribute `onehot_input`. ' +
                'Proceed with assumption that it receives one-hot encoded inputs.')
            self.onehot_input = True
        elif self.model.onehot_input is not None:
            assert self.model.onehot_input is True, (
                'Callback only works with models receiving one-hot encoded inputs.'
            )
            self.onehot_input = True

        if self.num_classes is None:
            self.num_classes:int = self.model.num_classes

        # Parse start_classes and end_classes
        if self.start_classes is None:
            self.start_classes = [label for label in range(self.num_classes)]
        if self.end_classes is None:
            self.end_classes = [label for label in range(self.num_classes)]

        self.nrows = len(self.start_classes)
        self.ncols = len(self.end_classes)

    def synthesize_images(self):
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            self.latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
                    
        x_synth = self.model.generator([self.latent_noise, self.label])
        x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def make_figure(self, x_synth:tf.Tensor, ratio:float):
        """Tile the synthetic images into a nice grid, then make and save a figure at
        the given interpolation ratio.
        
        Args:
            `x_synth`: A batch of synthetic images.
            `epoch`: Interpolation ratio.
        """
        # TODO:
        #   - Zero-padding currently only works for already normalized range.
        #   - Label for each category (it is obvious with MNIST, but might not
        #       for other datasets)
        #   - Update `cmap` for multi-channel pictures (eg CIFAR)
        fig, ax = subplots(constrained_layout=True, figsize=(self.ncols, 0.5 + self.nrows))
        fig.suptitle(f'{self.model.name} - Interpolation: {ratio*100:.2f}%')

        # Tile images into a grid
        x = x_synth
        x = tf.pad(x, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT', constant_values=1)
        x = tf.reshape(x, shape=[self.nrows, self.ncols, *x.shape[1:]])
        x = tf.concat(tf.unstack(x, axis=0), axis=1)
        x = tf.concat(tf.unstack(x, axis=0), axis=1)
        x = tf.image.crop_to_bounding_box(
            image=x,
            offset_height=1, offset_width=1,
            target_height=x.shape[0]-1, target_width=x.shape[1]-1
        )
        ax.axis('off')
        ax.imshow(x.numpy().squeeze(axis=-1), cmap='gray')
        fig.savefig(f"{self.path_png_folder}/{self.model.name}_itpl_{ratio:.4f}.png")
        close(fig)

class RepeatTensor(keras.layers.Layer):
    """Repeats the input based on given size.

    Typically, it is used for the discriminator in conditional GAN; spefically to
    repeat a multi-hot/one-hot vector to a stack of all-ones and and all-zeros
    images (before concatenating with real images).

    Args:
        `repeats`: Axes to repeats, right after the batch axis (0).
    """
    def __init__(self, repeats:List[int], **kwargs):
        """Initialize layer.
        
        Args:
            `repeats`: Axes to repeats, right after the batch axis (0).
        """        
        super().__init__(**kwargs)
        if any([not isinstance(item, int) for item in repeats]):
            raise TypeError(
                f"Expected a list or tuple of integers, got {type(repeats)}."
            )
        self.repeats = repeats
        self.reversed_repeats = self.repeats[::-1]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        return tf.TensorShape([input_shape[0], *self.repeats, input_shape[1]])

    def call(self, inputs):
        x = inputs
        for rp in self.reversed_repeats:
            x = tf.expand_dims(input=x, axis=1)
            x = tf.repeat(input=x, repeats=rp, axis=1)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"repeats": self.repeats})
        return config

if __name__ == '__main__':
    example = tf.constant([0, 1, 3])
    cat_encode = keras.layers.CategoryEncoding(num_tokens=4, output_mode='one_hot')
    repeat = RepeatTensor(repeats=[5, 5])
    
    x = example
    x = cat_encode(x)

    print('x:', x.numpy())
    x = repeat(x)
    print()
    print("RepeatTensor([5, 5])(x)[1]'s feature maps:\n", *[str(x[1, :, :, i].numpy())+'\n\n' for i in range(4)])

    print(x.shape)
    # print(*[[x[j, :, :, i].numpy().mean() for i in range(10)] for j in range(len(example))], sep='\n')
