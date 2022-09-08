from matplotlib.pyplot import rc, ion, subplots, show
import tensorflow as tf
keras = tf.keras

class PlotSyntheticCallback(keras.callbacks.Callback):
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