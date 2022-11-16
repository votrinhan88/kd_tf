import warnings
from typing import Callable, Union, List, Any, Literal
import os
import glob

import tensorflow as tf
keras = tf.keras
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import PIL

# TODO:
#   - Zero-padding currently only works for already normalized range.
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
                 save_freq:int=1,
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
        self.save_freq = save_freq
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
        if epoch % self.save_freq > 0:
            return

        fig, ax = plt.subplots(constrained_layout=True, figsize=(self.ncols, 0.5 + self.nrows))
        self.modify_suptitle(figure=fig, value=epoch)

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
        x = x.numpy()

        self.modify_axis(axis=ax)

        if self.image_dim[-1] == 1:
            ax.imshow(x.squeeze(axis=-1), cmap='gray')
        elif self.image_dim[-1] > 1:
            ax.imshow(x)

        fig.savefig(f"{self.path_png_folder}/{self.model.name}_epoch_{epoch:04d}.png")
        plt.close(fig)

    def modify_suptitle(self, figure:Figure, value:Union[int, float]):
        figure.suptitle(f'{self.model.name} - Epoch {value}')

    def modify_axis(self, axis:Axes):
        axis.axis('off')

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
        `class_names`: List of name of labels, should have length equal to total
            number of classes. Leave as `None` for generic `'class x'` names.
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
                 class_names:Union[None, List[str]]=None,
                 onehot_input:Union[None, bool]=None,
                 keep_noise:bool=True,
                 delete_png:bool=True,
                 save_freq:int=1,
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
            `class_names`: List of name of labels, should have length equal to total
                number of classes. Leave as `None` for generic `'class x'` names.
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
            filename=filename,
            nrows=None,
            ncols=None,
            postprocess_fn=postprocess_fn,
            latent_dim=latent_dim,
            image_dim=image_dim,
            keep_noise=keep_noise,
            delete_png=delete_png,
            save_freq=save_freq,
            duration=duration,
            **kwargs
        )
        self.target_classes = target_classes
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.class_names = class_names
        self.onehot_input = onehot_input

    def handle_args(self):
        super(MakeConditionalSyntheticGIFCallback, self).handle_args()
        if self.num_classes is None:
            self.num_classes:int = self.model.num_classes

        if self.class_names is None:
            self.class_names = [{f'Class {i}' for i in range(self.num_classes)}]

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

    def modify_axis(self, axis:Axes):
        xticks = (self.image_dim[1] + 1)*np.arange(len(self.target_classes)) + self.image_dim[1]/2
        xticklabels = [self.class_names[label] for label in self.target_classes]
        
        axis.set_frame_on(False)
        axis.tick_params(axis='both', length=0)
        axis.set(yticks=[], xticks=xticks, xticklabels=xticklabels)

# TODO: custom image label on ax.xticks
class MakeInterpolateSyntheticGIFCallback(MakeSyntheticGIFCallback):
    # TODO: Spherical linear interpolation
    """Callback to generate synthetic images, interpolated between the classes of a
    Conditional Generative Adversarial Network.
    
    The callback can only work with models receiving one-hot encoded inputs. It
    will make figures at the end of the last epoch.
    
    Args:
        `filename`: Path to save GIF to. Defaults to `'./logs/GAN_itpl.gif'`.
        `start_classes`: Classes at the start of interpolation along the rows, leave
            as `None` to include all classes. Defaults to `None`.
        `stop_classes`: Classes at the stop of interpolation along the columns, leave
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
        `class_names`: List of name of labels, should have length equal to total
            number of classes. Leave as `None` for generic `'class x'` names.
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
                 stop_classes:List[int]=None,
                 num_itpl:int=51,
                 itpl_method:Literal['linspace', 'slerp']='linspace',
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 num_classes:Union[None, int]=None,
                 class_names:Union[None, List[str]]=None,
                 keep_noise:bool=True,
                 delete_png:bool=True,
                 duration:float=5000,
                 **kwargs):
        """Initialize callback.
        
        Args:
            `filename`: Path to save GIF to. Defaults to `'./logs/GAN_itpl.gif'`.
            `start_classes`: Classes at the start of interpolation along the rows, leave
                as `None` to include all classes. Defaults to `None`.
            `stop_classes`: Classes at the stop of interpolation along the columns, leave
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
            `class_names`: List of name of labels, should have length equal to total
                number of classes. Leave as `None` for generic `'class x'` names.
                Defaults to `None`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete PNG files and folder at `filename/png` after
                training. Defaults to `True`.
            `duration`: Duration of the generated GIF in milliseconds.
                Defaults to `5000`.
        """
        assert num_itpl > 2, (
            '`num_interpolate` (including the left and right classes) must be' +
            ' larger than 2.'
        )
        assert itpl_method in ['linspace', 'slerp'], (
            "`itpl_method` must be 'linspace' or 'slerp'"
        )

        super().__init__(
            filename=filename,
            nrows=None,
            ncols=None,
            postprocess_fn=postprocess_fn,
            latent_dim=latent_dim,
            image_dim=image_dim,
            keep_noise=keep_noise,
            delete_png=delete_png,
            duration=duration,
            **kwargs
        )
        self.itpl_method = itpl_method
        self.start_classes = start_classes
        self.stop_classes = stop_classes
        self.num_itpl = num_itpl
        self.num_classes = num_classes
        # Reset unused inherited attributes
        self.save_freq = None

    def on_epoch_end(self, epoch, logs=None):
        # Deactivate MakeSyntheticGIFCallback.on_epoch_end()
        return keras.callbacks.Callback.on_epoch_end(self, epoch, logs)
    
    def on_train_end(self, logs=None):
        # Interpolate from start- to stop-classes
        itpl_ratios = tf.cast(tf.linspace(start=0, stop=1, num=self.num_itpl), dtype=tf.float32).numpy().tolist()
        for ratio in itpl_ratios:
            label = self._interpolate(start=self.start, stop=self.stop, ratio=ratio)
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

        # Parse interpolate method, start_classes and stop_classes
        if self.itpl_method == 'linspace':
            self._interpolate = self.linspace
        elif self.itpl_method == 'slerp':
            self._interpolate = self.slerp

        if self.start_classes is None:
            self.start_classes = [label for label in range(self.num_classes)]
        if self.stop_classes is None:
            self.stop_classes = [label for label in range(self.num_classes)]

        self.nrows = len(self.start_classes)
        self.ncols = len(self.stop_classes)

    def synthesize_images(self):
        if self.keep_noise is False:
            batch_size = self.nrows*self.ncols
            self.latent_noise = tf.random.normal(shape=[batch_size, self.latent_dim])
                    
        x_synth = self.model.generator([self.latent_noise, self.label])
        x_synth = self.postprocess_fn(x_synth)
        return x_synth

    def modify_suptitle(self, figure:Figure, value:Union[int, float]):
        figure.suptitle(f'{self.model.name} - {self.itpl_method} interpolation: {value*100:.2f}%')

    def modify_axis(self, axis:Axes):
        xticks = (self.image_dim[1] + 1)*np.arange(len(self.stop_classes)) + self.image_dim[1]/2
        xticklabels = [self.class_names[label] for label in self.stop_classes]

        yticks = (self.image_dim[1] + 1)*np.arange(len(self.start_classes)) + self.image_dim[1]/2
        yticklabels = [self.class_names[label] for label in self.start_classes]
        
        axis.set_frame_on(False)
        axis.tick_params(axis='both', length=0)
        axis.set(
            xlabel='Stop classes', xticks=xticks, xticklabels=xticklabels,
            ylabel='Start classes', yticks=xticks, yticklabels=yticklabels)

    def precompute_inputs(self):
        super(MakeInterpolateSyntheticGIFCallback, self).precompute_inputs()
        # Convert to one-hot labels
        start = tf.one_hot(indices=self.start_classes, depth=self.num_classes)
        stop = tf.one_hot(indices=self.stop_classes, depth=self.num_classes)

        # Expand dimensions to have shape [nrows, ncols, num_classes]
        start = tf.expand_dims(input=start, axis=1)
        start = tf.repeat(start, repeats=self.ncols, axis=1)
        stop = tf.expand_dims(input=stop, axis=0)
        stop = tf.repeat(stop, repeats=self.nrows, axis=0)

        self.start = start
        self.stop = stop

        if self.itpl_method == 'slerp':
            # Normalize (L2) to [-1, 1] for numerical stability
            norm_start = start/tf.norm(start, axis=-1)
            norm_stop = stop/tf.norm(stop, axis=-1)

            dotted = tf.math.reduce_sum(norm_start*norm_stop, axis=-1)
            # Clip to [-1, 1] for numerical stability
            clipped = tf.clip_by_value(dotted, -1, 1)
            omegas = tf.acos(clipped)
            sinned = tf.sin(omegas)

            # Expand dimensions to have shape [nrows, ncols, num_classes]
            omegas = tf.expand_dims(omegas, axis=-1)
            omegas = tf.repeat(omegas, repeats=self.num_classes, axis=-1)
            sinned = tf.expand_dims(sinned, axis=-1)
            sinned = tf.repeat(sinned, repeats=self.num_classes, axis=-1)
            zeros_mask = (omegas == 0)

            self.omegas = omegas
            self.sinned = sinned
            self.zeros_mask = zeros_mask

    def linspace(self, start, stop, ratio:float):
        label = ((1-ratio)*start + ratio*stop)
        return label
    
    def slerp(self, start, stop, ratio:float):
        label = tf.where(
            self.zeros_mask,
            # Normal case: omega(s) != 0
            self.linspace(start=start, stop=stop, ratio=ratio),
            # Special case: omega(s) == 0 --> Use L'Hospital's rule for sin(0)/0
            (
                tf.sin((1-ratio)*self.omegas) / self.sinned * start + 
                tf.sin(ratio    *self.omegas) / self.sinned * stop
            )
        )
        return label

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
    print("RepeatTensor([5, 5])(x)[1]'s feature maps:\n",
          *[str(x[1, :, :, i].numpy())+'\n\n' for i in range(4)])

    print(x.shape)
    # print(*[[x[j, :, :, i].numpy().mean() for i in range(10)] for j in range(len(example))], sep='\n')
