
import hypertune
import tensorflow as tf

hpt = hypertune.HyperTune()

class HPTCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global hpt
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="val_accuracy",
            metric_value=logs["val_accuracy"],
            global_step=epoch,
        )
