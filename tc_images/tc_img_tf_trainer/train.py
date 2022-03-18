

import pickle
import subprocess
import sys
import fire
import pandas as pd
import tensorflow as tf
import datetime
import os

CSV_COLUMNS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

LABEL_COLUMN = "Churn"

DEFAULTS = [['na'], ['na'], ['na'], ['na'], [0.0], ['na'], ['na'], ['na'],
            ['na'], ['na'], ['na'], ['na'], ['na'], ['na'], ['na'], 
            ['na'], ['na'], [0.0], [0.0], ['na']]

AIP_MODEL_DIR = os.environ["AIP_MODEL_DIR"]

def features_and_labels(row_data):
    cols = tf.io.decode_csv(row_data, record_defaults=DEFAULTS)
    feats = {
        'gender': tf.reshape(cols[0], [1,]),
        'SeniorCitizen': tf.reshape(cols[1],[1,]),
        'Partner': tf.reshape(cols[2],[1,]),
        'Dependents': tf.reshape(cols[3],[1,]),
        'tenure': tf.reshape(cols[4],[1,]),
        'PhoneService': tf.reshape(cols[5],[1,]),
        'MultipleLines': tf.reshape(cols[6],[1,]),
        'InternetService': tf.reshape(cols[7],[1,]),
        'OnlineSecurity': tf.reshape(cols[8],[1,]),
        'OnlineBackup': tf.reshape(cols[9],[1,]),
        'DeviceProtection': tf.reshape(cols[10],[1,]),
        'TechSupport': tf.reshape(cols[11],[1,]),
        'StreamingTV': tf.reshape(cols[12],[1,]),
        'StreamingMovies': tf.reshape(cols[13],[1,]),
        'Contract': tf.reshape(cols[14],[1,]),
        'PaperlessBilling': tf.reshape(cols[15],[1,]),
        'PaymentMethod': tf.reshape(cols[16],[1,]),
        'MonthlyCharges': tf.reshape(cols[17],[1,]),
        'TotalCharges': tf.reshape(cols[18],[1,]),
        'Churn': cols[19]
    }
    label = feats.pop('Churn')
    label_int = tf.case([(tf.math.equal(label,tf.constant(['No'])), lambda: 0),
                        (tf.math.equal(label,tf.constant(['Yes'])), lambda: 1)])
    
    return feats, label_int

def load_dataset(pattern, batch_size=1, mode='eval'):
    # Make a CSV dataset
    filelist = tf.io.gfile.glob(pattern)
    dataset = tf.data.TextLineDataset(filelist).skip(1)
    dataset = dataset.map(features_and_labels)

    # Shuffle and repeat for training
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=10*batch_size).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(10)

    return dataset

def train_evaluate(training_dataset_path, validation_dataset_path, batch_size, num_train_examples, num_evals):
    inputs = {
        'gender': tf.keras.layers.Input(name='gender',shape=[None],dtype='string'),
        'SeniorCitizen': tf.keras.layers.Input(name='SeniorCitizen',shape=[None],dtype='string'),
        'Partner': tf.keras.layers.Input(name='Partner',shape=[None],dtype='string'),
        'Dependents': tf.keras.layers.Input(name='Dependents',shape=[None],dtype='string'),
        'tenure': tf.keras.layers.Input(name='tenure',shape=[None],dtype='int32'),
        'PhoneService': tf.keras.layers.Input(name='PhoneService',shape=[None],dtype='string'),
        'MultipleLines': tf.keras.layers.Input(name='MultipleLines',shape=[None],dtype='string'),
        'InternetService': tf.keras.layers.Input(name='InternetService',shape=[None],dtype='string'),
        'OnlineSecurity': tf.keras.layers.Input(name='OnlineSecurity',shape=[None],dtype='string'),
        'OnlineBackup': tf.keras.layers.Input(name='OnlineBackup',shape=[None],dtype='string'),
        'DeviceProtection': tf.keras.layers.Input(name='DeviceProtection',shape=[None],dtype='string'),
        'TechSupport': tf.keras.layers.Input(name='TechSupport',shape=[None],dtype='string'),
        'StreamingTV': tf.keras.layers.Input(name='StreamingTV',shape=[None],dtype='string'),
        'StreamingMovies': tf.keras.layers.Input(name='StreamingMovies',shape=[None],dtype='string'),
        'Contract': tf.keras.layers.Input(name='Contract',shape=[None],dtype='string'),
        'PaperlessBilling': tf.keras.layers.Input(name='PaperlessBilling',shape=[None],dtype='string'),
        'PaymentMethod': tf.keras.layers.Input(name='PaymentMethod',shape=[None],dtype='string'),
        'MonthlyCharges': tf.keras.layers.Input(name='MonthlyCharges',shape=[None],dtype='float'),
        'TotalCharges': tf.keras.layers.Input(name='TotalCharges',shape=[None],dtype='float')
    }
    
    batch_size = int(batch_size)
    num_train_examples = int(num_train_examples)
    num_evals = int(num_evals)
    
    feat_cols = {
        'tenure': tf.feature_column.numeric_column('tenure'),
        'TotalCharges': tf.feature_column.numeric_column('TotalCharges'),
        'MonthlyCharges': tf.feature_column.numeric_column('MonthlyCharges'),
        'SeniorCitizen': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='SeniorCitizen', hash_bucket_size=3
            )
        ),
        'gender': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='gender', hash_bucket_size=2
            )
        ),
        'Partner': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='Partner', hash_bucket_size=2
            )
        ),
        'Dependents': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='Dependents', hash_bucket_size=2
            )
        ),
        'PhoneService': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='PhoneService', hash_bucket_size=2
            )
        ),
        
        'MultipleLines': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='MultipleLines', hash_bucket_size=3
            )
        ),
        'InternetService': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='InternetService', hash_bucket_size=3
            )
        ),
        'OnlineSecurity': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='OnlineSecurity', hash_bucket_size=3
            )
        ),
        'OnlineBackup': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='OnlineBackup', hash_bucket_size=3
            )
        ),
        'DeviceProtection': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='DeviceProtection', hash_bucket_size=3
            )
        ),
        'TechSupport': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='TechSupport', hash_bucket_size=3
            )
        ),
        'StreamingTV': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='StreamingTV', hash_bucket_size=3
            )
        ),
        'StreamingMovies': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='StreamingMovies', hash_bucket_size=3
            )
        ),
        'Contract': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='Contract', hash_bucket_size=3
            )
        ),
        'PaperlessBilling': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='PaperlessBilling', hash_bucket_size=2
            )
        ),
        'PaymentMethod': tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key='PaymentMethod', hash_bucket_size=3
            )
        )
    }
    
    dnn_inputs = tf.keras.layers.DenseFeatures(
        feature_columns=feat_cols.values())(inputs)
    h1 = tf.keras.layers.Dense(64, activation='relu')(dnn_inputs)
    h2 = tf.keras.layers.Dense(128, activation='relu')(h1)
    h3 = tf.keras.layers.Dense(64, activation='relu')(h2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(h3)
    
    model = tf.keras.models.Model(inputs=inputs,outputs=output)
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    trainds = load_dataset(
        pattern=training_dataset_path,
        batch_size=batch_size,
        mode='train')
    
    evalds = load_dataset(
        pattern=validation_dataset_path,
        mode='eval')
    
    
    steps_per_epoch = num_train_examples // (batch_size * num_evals)
    
    history = model.fit(
        trainds,
        validation_data=evalds,
        validation_steps=100,
        epochs=num_evals,
        steps_per_epoch=steps_per_epoch
    )

    #model_export_path = os.path.join(AIP_MODEL_DIR, "savedmodel")
    model_export_path = os.path.join(AIP_MODEL_DIR)
    tf.saved_model.save(
        obj=model, export_dir=model_export_path)  # with default serving function
    
    print("Exported trained model to {}".format(model_export_path))
    
if __name__ == '__main__':
    fire.Fire(train_evaluate)
