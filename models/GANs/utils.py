from typing import Callable, Union, List, Any

from matplotlib.pyplot import rc, ion, subplots, show, close
import tensorflow as tf
keras = tf.keras
import PIL
import os
import glob

class PlotSyntheticCallback(keras.callbacks.Callback):
    ver:int = 2
    
    def __init__(self, num_epochs:int, num_synthetic:int=5):
        super().__init__()
        self.num_epochs     = num_epochs
        self.num_synthetic  = num_synthetic
        
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 12

        rc('font', size=SMALL_SIZE)          # controls default text sizes
        rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    def on_train_begin(self, logs=None):
        ion()
        self.fig, self.ax = subplots(
            nrows=self.num_synthetic,
            ncols=self.num_epochs,
            sharex='all',
            sharey='all',
            squeeze=False,
            constrained_layout=True)
        self.fig.suptitle(self.model.name)
        self.ax[0, 0].set(
            xticks=[],
            xticklabels=[],
            yticks=[],
            yticklabels=[])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.show()

    def on_epoch_end(self, epoch:int, logs=None):
        noise = tf.random.normal(shape=[self.num_synthetic, self.model.latent_dim[0]], seed=17)
        img = self.model.generator(noise)
        pred = self.model.discriminator(img)

        img = tf.reshape(img, (-1, 28, 28))
        for col in range(self.num_synthetic):
            self.ax[col, epoch].imshow(img[col], cmap='gray')
            self.ax[col, epoch].set(title=f'{float(tf.nn.sigmoid(pred[col]))*100:.2f}%')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.show()

    def on_train_end(self, logs=None):
        show(block=True)

class MakeSyntheticGIFCallback(keras.callbacks.Callback):
    # TODO: Update `cmap` in `on_epoch_end` for multi-channel pictures (eg CIFAR)
    """Callback to generate synthetic images, typically used with a Generative
    Adversarial Network.
    
    Args:
        `nrows`: Number of rows in subplot figure. Defaults to `1`.
        `ncols`: Number of columns in subplot figure. Defaults to `5`.
        `postprocess_fn`: Post-processing function to map synthetic images back to
            the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
            Defaults to `None`.
        `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
            model. Defaults to `None`.
        `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
            from model. Defaults to `None`.
        `path`: Path to save GIF to. Defaults to `'./logs'`.
        `keep_noise`: Flag to feed the same latent noise to generator for the whole
            training to monitor progress easier. Defaults to `True`.
        `delete_png`: Flag to delete files and folder at `path/png` after training.
            Defaults to `True`.
        `duration`: Duration of the generated GIF in seconds. Defaults to `5.0`.
    """
    def __init__(self,
                 nrows:int=1,
                 ncols:int=5,
                 postprocess_fn:Union[None, Callable[[Any], Any]]=None,
                 latent_dim:Union[None, int]=None,
                 image_dim:Union[None, List[int]]=None,
                 path:str='./logs',
                 keep_noise:bool=True,
                 delete_png:bool=True,
                 duration:float=5.0,
                 **kwargs):
        """Initialize callback.
        
        Args:
            `nrows`: Number of rows in subplot figure. Defaults to `1`.
            `ncols`: Number of columns in subplot figure. Defaults to `5`.
            `postprocess_fn`: Post-processing function to map synthetic images back to
                the plot range, ideally [0, 1]. Leave as `None` to skip post-processing.  
                Defaults to `None`.
            `latent_dim`: Dimension of latent space, leave as `None` to be parsed from
                model. Defaults to `None`.
            `image_dim`: Dimension of synthetic images, leave as `None` to be parsed
                from model. Defaults to `None`.
            `path`: Path to save GIF to. Defaults to `'./logs'`.
            `keep_noise`: Flag to feed the same latent noise to generator for the whole
                training. Defaults to `True`.
            `delete_png`: Flag to delete files and folder at `path/png` after training.
                Defaults to `True`.
            `duration`: Duration of the generated GIF in seconds. Defaults to `5.0`.
        """
        super().__init__(**kwargs)
        self.nrows = nrows
        self.ncols = ncols
        self.postprocess_fn = postprocess_fn
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.path = path
        self.keep_noise = keep_noise
        self.delete_png = delete_png
        self.duration = duration

    def on_train_begin(self, logs=None):
        # Handle postprocessing function
        if self.postprocess_fn is None:
            self.postprocess_fn = lambda x:x
        # Handle latent and image dimension
        if self.latent_dim is None:
            self.latent_dim:int = self.model.latent_dim
        if self.image_dim is None:
            self.image_dim:int = self.model.image_dim
        # Pre-allocate latent noise
        self.latent_noise = None
        # Renew/create folder containing PNG files
        if os.path.isdir(f'{self.path}/png'):
            for png in glob.glob(f'{self.path}/png/*.png'):
                os.remove(png)
        else:
            os.mkdir(f'{self.path}/png')
        return super().on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        # Generate new images (use existing latent noise if specified)
        if (self.latent_noise is None) or (self.keep_noise is False):
            batch_size = self.nrows*self.ncols
            self.latent_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        x_synth = self.model.generator(self.latent_noise)
        x_synth = self.postprocess_fn(x_synth)
        # Create and save figure
        fig, ax = subplots(
            nrows=self.nrows, ncols=self.ncols,
            sharex='all', sharey='all', squeeze=False,
            constrained_layout=True, figsize=(self.ncols, 0.5 + self.nrows))
        fig.suptitle(f'{self.model.name} - Epoch {epoch}')
        for row in range(self.nrows):
            for col in range(self.ncols):
                idx = row*self.nrows + col
                ax[row, col].set(
                    xticks=[],
                    xticklabels=[],
                    yticks=[],
                    yticklabels=[])
                ax[row, col].imshow(x_synth[idx].numpy().reshape(self.image_dim).squeeze(axis=-1), cmap='gray')
        fig.savefig(f"./logs/png/{self.model.name}_epoch_{epoch:04d}.png")
        close(fig)
        return super().on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        # Make GIF
        path_png = f'{self.path}/png/*.png'
        path_gif = f'{self.path}/{self.model.name}.gif'
        img, *imgs = [PIL.Image.open(f) for f in sorted(glob.glob(path_png))]
        img.save(
            fp=path_gif,
            format='GIF',
            append_images=imgs,
            save_all=True,
            duration=self.duration/(len(imgs) + 1),
            loop=0)

        if self.delete_png is True:
            for png in glob.glob(f'{self.path}/png/*.png'):
                os.remove(png)
            os.rmdir(f'{self.path}/png')

        return super().on_train_end(logs)