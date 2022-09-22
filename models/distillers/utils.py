import numpy as np
import collections
import csv
from keras.utils import io_utils
import tensorflow as tf
keras = tf.keras
from tensorflow.python.platform import tf_logging as logging


class CSVLogger_custom(keras.callbacks.CSVLogger):
    """Re-implementation of keras.callbacks.CSVLogger to use with model.fit() when
    validation_freq > 1.
    
    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.
    """
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                isinstance(k, collections.abc.Iterable)
                and not is_zero_dim_ndarray
            ):
                return '"[%s]"' % (", ".join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict(
                (k, logs[k]) if k in logs else (k, "NA") for k in self.keys
            )

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": epoch})
        row_dict.update((key, handle_value(logs[key])) if key in logs else (key, "") for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

class ThresholdStopping(keras.callbacks.Callback):
    # TODO: add parameter epoch_begin: epoch to begin threshold stopping (ignore
    # before that)
    """Stop training when a monitored metric is over/under a certain threshold.
    
    Assuming the goal of a training is to assure the loss stays under a threshold.
    With this, the metric to be monitored would be `'loss'`, and mode would be
    `'min'`. A `model.fit()` training loop will compare at end of every epoch
    whether the loss is below the given threshold, considering the `'patience'` if
    applicable. Once it's found to be under the threshold, `model.stop_training`
    is marked True and the training terminates. The quantity to be monitored needs
    to be available in `logs` dict. To make it so, pass the loss or metrics at
    `model.compile()`.
    
    Args:
      monitor: Quantity to be monitored.
      threshold: The threshold to compare monitored quantity with.
      patience: Number of epochs violating the threshold after which training will
          be stopped.
      epoch_begin: The epoch from which the monitoring begins (epochs before this
          are ignored).
      mode: One of `{'auto', 'over', 'under'}`. In `'over'` mode, training will stop
          when the quantity monitored is larger than the threshold; in `'under'`
          mode it will stop when the quantity monitored is lower than the threshold;
          in `'auto'` mode, the direction is automatically inferred from the name of
          the monitored quantity.
      verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
          messages when the callback takes an action.
      restore_best_weights: Whether to restore model weights from the epoch with the
          best value of the monitored quantity. If False, the model weights obtained
          at the last step of training are used.
    """

    def __init__(
        self,
        threshold:float=0.1,
        monitor:str='val_loss',
        patience:int=0,
        epoch_begin:int=0,
        verbose:int=0,
        mode:str='auto',
    ):
        super().__init__()

        self.threshold = threshold
        self.monitor = monitor
        self.patience = patience
        self.epoch_begin = epoch_begin
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0

        if mode not in ['auto', 'over', 'under']:
            logging.warning(f"ThresholdStopping mode {mode} is unknown, fallback to auto mode.")
            mode = 'auto'

        if mode == 'over':
            self.monitor_op = np.less
        elif mode == 'under':
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        self.wait += 1
        if self._is_complying(current):
            self.wait = 0

        # Only check after the specified epoch (default 0).
        if self.wait >= self.patience and epoch > self.epoch_begin:
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f"Epoch {self.stopped_epoch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                f"Threshold stopping conditioned on metric `{self.monitor}` " +
                f"which is not available. Available metrics are: {','.join(list(logs.keys()))}",
            )
        return monitor_value

    def _is_complying(self, monitor_value):
        return self.monitor_op(monitor_value, self.threshold)

class ValidationFreqPrinter(keras.callbacks.Callback):
    """Print validation results only on epoch with validation step. Typically used
    when calling `model.fit()` with `validation_freq` > 1.
    """

    def __init__(self, validation_freq:int=1):
        super().__init__()
        self.verbose = 1
        self.validation_freq = validation_freq

    def on_epoch_end(self, epoch, logs=None):
        if self.model._should_eval(epoch, self.validation_freq) == True:
            print(f"Epoch {epoch}: {', '.join(f'{k}: {v:.4f}' for k, v in logs.items())}")