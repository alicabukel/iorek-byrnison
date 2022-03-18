import tensorflow as tf
import json
import logging

from trainer.constants import UNWANTED_COLS, LABEL_COLUMN, CSV_COLUMNS, DEFAULTS

logging.info(tf.version.VERSION)

def features_and_labels(row_data):
    for unwanted_col in UNWANTED_COLS:
        row_data.pop(unwanted_col)
    label = row_data.pop(LABEL_COLUMN)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(["Yes", "No"]),
        values=tf.constant([1, 0], dtype=tf.int64))
    table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)
    label = table[label]
    return row_data, label

def load_dataset(pattern, batch_size, num_repeat):
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=pattern,
        batch_size=batch_size,
        column_names=CSV_COLUMNS,
        column_defaults=DEFAULTS,
        num_epochs=num_repeat,
        na_value=" ",
        shuffle_buffer_size=1000000,
    )
    return dataset.map(features_and_labels)

def create_train_dataset(pattern, batch_size):
    dataset = load_dataset(pattern, batch_size, num_repeat=None)
    return dataset.prefetch(1)

def create_eval_dataset(pattern, batch_size):
    dataset = load_dataset(pattern, batch_size, num_repeat=1)
    return dataset.prefetch(1)
