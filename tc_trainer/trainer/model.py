
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf

from trainer.transform import transform
from trainer.constants import NUM_COLS, CAT_COLS

def build_dnn_model(nbuckets, nnsize, lr):
    inputs = {
        colname: layers.Input(name=colname, shape=(), dtype="float32")
        for colname in NUM_COLS
    }
    inputs.update(
        {
            colname: layers.Input(name=colname, shape=(), dtype="string")
            for colname in CAT_COLS
        }
    )

    # transforms
    transformed, feature_columns = transform(inputs, NUM_COLS, nbuckets)
    dnn_inputs = layers.DenseFeatures(feature_columns.values())(transformed)

    x = dnn_inputs
    for layer, nodes in enumerate(nnsize):
        x = layers.Dense(nodes, activation="relu", name=f"h{layer}")(x)
    output = layers.Dense(1, name="churn")(x)

    model = models.Model(inputs, output)

    lr_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    model.compile(optimizer=lr_optimizer, loss=loss, metrics=['accuracy'])

    return model
