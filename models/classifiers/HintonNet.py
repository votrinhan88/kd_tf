from typing import List

import tensorflow as tf
keras = tf.keras

class HintonNet(keras.Model):
    """Baseline model in implemented in paper 'Distilling the Knowledge in a Neural
    Network' - Hinton et al. (2015), DOI: 10.48550/arXiv.1503.02531. Originally
    described in Improving neural networks by preventing co-adaptation of
    feature detectors - Hinton et al (2012), DOI: 10.48550/arXiv.1207.0580

    Args:
        `input_dim`: Dimension of input images. Defaults to `[28, 28, 1]`.
        `hidden_layers`: Number of nodes in each hidden layer.
            Defaults to `[1200, 1200]`.
        `num_classes`: Number of output nodes. Defaults to `10`.
        `return_logits`: Flag to choose between return logits or probability.
            Defaults to `False`.
    
    Kwargs:
        Additional keyword arguments passed to `keras.Model.__init__`.

    Two versions:
    - Teacher: `hidden_layers` = [1200, 1200]
    - Student: `hidden_layers` = [800, 800]
    """    
    _name = 'HintonNet'

    def __init__(self,
                 input_dim:List[int]=[28, 28, 1],
                 hidden_layers:List[int]=[1200, 1200],
                 num_classes:int=10,
                 return_logits:bool=False,
                 **kwargs):
        """Initialize model.
        
        Args:
            `input_dim`: Dimension of input images. Defaults to `[28, 28, 1]`.
            `hidden_layers`: Number of nodes in each hidden layer.
                Defaults to `[1200, 1200]`.
            `num_classes`: Number of output nodes. Defaults to `10`.
            `return_logits`: Flag to choose between return logits or probability.
                Defaults to `False`.
        
        Kwargs:
            Additional keyword arguments passed to `keras.Model.__init__`.
        """
        assert isinstance(return_logits, bool), '`return_logits` must be of type bool.'

        super().__init__(self, name=self._name, **kwargs)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.return_logits = return_logits

        self.weight_constraint = keras.constraints.MaxNorm(max_value=15)

        self.flatten    = keras.layers.Flatten()
        self.dropout_in = keras.layers.Dropout(rate=0.2)
        self._hidden_layers = []
        for num_nodes in self.hidden_layers:
            self._hidden_layers.extend([
                keras.layers.Dense(units=num_nodes, kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01), kernel_constraint=self.weight_constraint),
                keras.layers.Activation(tf.nn.relu),
                keras.layers.Dropout(rate=0.5),
            ])

        if self.return_logits is False:
            if self.num_classes == 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.sigmoid, kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01), kernel_constraint=self.weight_constraint)
            elif self.num_classes > 1:
                self.pred = keras.layers.Dense(units=self.num_classes, name='pred', activation=tf.nn.softmax, kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01), kernel_constraint=self.weight_constraint)
        elif self.return_logits is True:
            self.logits = keras.layers.Dense(units=self.num_classes, name='logits', kernel_initializer=keras.initializers.RandomNormal(mean=0, stddev=0.01), kernel_constraint=self.weight_constraint)

    def call(self, inputs, training:bool=False):
        x = self.flatten(inputs)
        x = self.dropout_in(x)
        for layer in self._hidden_layers:
            x = layer(x)
        if self.return_logits is False:
            x = self.pred(x)
        elif self.return_logits is True:
            x = self.logits(x)
        return x

    def build(self):
        super().build(input_shape=[None, *self.input_dim])

    def summary(self, as_functional:bool=False, **kwargs):
        """Prints a string summary of the network.

        Args:
            `as_functional`: Flag to print from a dummy functional model.
                Defaults to `False`.

        Kwargs:
            Additional keyword arguments passed to `keras.Model.summary`.
        """
        inputs = keras.layers.Input(shape=self.input_dim)
        outputs = self.call(inputs)

        if as_functional is True:
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
        input_dim=[28, 28, 1],
        hidden_layers=[1200, 1200],
        num_classes=10
    )
    net.build()
    net.summary(as_functional=True, expand_nested=True, line_length=120)
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