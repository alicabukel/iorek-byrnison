
import os
import logging
import tensorflow as tf
from tensorflow.keras import callbacks

from trainer.data import create_train_dataset, create_eval_dataset
from trainer.model import build_dnn_model
from trainer.tuner import HPTCallback

def train_and_evaluate(hparams, hptune=False):
    batch_size = hparams["batch_size"]
    nbuckets = hparams["nbuckets"]
    lr = hparams["lr"]
    nnsize = [int(s) for s in hparams["nnsize"].split()]
    eval_data_path = hparams["eval_data_path"]
    num_evals = hparams["num_evals"]
    num_examples_to_train_on = hparams["num_examples_to_train_on"]
    output_dir = hparams["output_dir"]
    train_data_path = hparams["train_data_path"]

    model_export_path = os.path.join(output_dir, "savedmodel")
    checkpoint_path = os.path.join(output_dir, "checkpoints")
    tensorboard_path = os.path.join(output_dir, "tensorboard")

    if tf.io.gfile.exists(output_dir):
        tf.io.gfile.rmtree(output_dir)

    model = build_dnn_model(nbuckets, nnsize, lr)
    logging.info(model.summary())

    trainds = create_train_dataset(train_data_path, batch_size)
    evalds = create_eval_dataset(eval_data_path, batch_size)

    steps_per_epoch = num_examples_to_train_on // (batch_size * num_evals)

    checkpoint_cb = callbacks.ModelCheckpoint(
        checkpoint_path, save_weights_only=True, verbose=1
    )
    tensorboard_cb = callbacks.TensorBoard(tensorboard_path, histogram_freq=1)
    callbacks_list= [checkpoint_cb, tensorboard_cb]
    if hptune:
        callbacks_list.append(HPTCallback())
        
    history = model.fit(
        trainds,
        validation_data=evalds,
        epochs=num_evals,
        steps_per_epoch=max(1, steps_per_epoch),
        verbose=2,
        callbacks=callbacks_list,
    )

    model.save(model_export_path)
    return history
