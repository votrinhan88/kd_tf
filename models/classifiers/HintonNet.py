import tensorflow as tf
keras = tf.keras

class HintonNet(keras.Model):
    """Baseline model in implemented in paper 'Distilling the Knowledge in a Neural
    Network' - Hinton et al. (2015)
    DOI: 10.48550/arXiv.1503.02531

    Args:
        `num_inputs`: Number of input nodes. Defaults to `784`.
        `num_hiddens`: Number of nodes in each hidden layer. Defaults to `1200`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `False`.

    Two versions:
    - Teacher: 1200 nodes in each of two hidden layers
    - Student: 800 nodes in each of two hidden layers
    """    
    _name = 'HintonNet'

    def __init__(self,
                 num_inputs:int=784,
                 num_hiddens:int=1200,
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        """Initialize model.
        
        Args:
            `num_inputs`: Number of input nodes. Defaults to `784`.
            `num_hiddens`: Number of nodes in each hidden layer. Defaults to `1200`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `False`.
        """
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super().__init__(self, name=self._name, **kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.dense_1      = keras.layers.Dense(units=self.num_hiddens)
        self.leaky_relu_1 = keras.layers.LeakyReLU(alpha=0.3)
        self.dense_2      = keras.layers.Dense(units=self.num_hiddens)
        self.leaky_relu_2 = keras.layers.LeakyReLU(alpha=0.3)

        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits')

    def call(self, inputs, training:bool=False):
        x = self.dense_1(inputs)
        x = self.leaky_relu_1(x)
        x = self.dense_2(x)
        x = self.leaky_relu_2(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def build(self):
        super().build(input_shape=[None, self.num_inputs])

    def summary(self, with_graph:bool=False, **kwargs):
        inputs = keras.layers.Input(shape=self.num_inputs)
        outputs = self.call(inputs)

        if with_graph is True:
            dummy_model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
            dummy_model.summary(**kwargs)
        else:
            super().summary(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_inputs': self.num_inputs,
            'num_hiddens': self.num_hiddens,
            'num_classes': self.num_classes,
            'return_logits': self.return_logits
        })
        return config

if __name__ == '__main__':
    # Change path
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)
    
    from dataloader import dataloader

    ds = dataloader(
        dataset='mnist',
        rescale=[-1, 1],
        batch_size_train=64,
        batch_size_test=1024
    )

    net = HintonNet(
        input_dim=[32, 32, 1],
        num_classes=10
    )
    net.build()
    net.summary()
    net.compile(
        metrics=['accuracy'], 
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy())

    best_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'./logs/{net.name}_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
    )
    csv_logger = keras.callbacks.CSVLogger(
        filename=f'./logs/{net.name}.csv',
        append=True
    )

    net.fit(
        ds['train'],
        epochs=10,
        callbacks=[best_callback, csv_logger],
        shuffle=True,
        validation_data=ds['test']
    )